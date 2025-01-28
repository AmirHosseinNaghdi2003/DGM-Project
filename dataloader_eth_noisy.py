# dataloader_eth.py

import os
import math
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset

def seq_collate(data):
    """
    Collate function to stack batched data.

    Args:
        data (list): List of dictionaries with 'past_traj' and 'future_traj'.

    Returns:
        dict: Dictionary with batched 'past_traj' and 'future_traj'.
    """
    past_traj = [item['past_traj'] for item in data]
    future_traj = [item['future_traj'] for item in data]

    # Stack tensors along the batch dimension
    past_traj = torch.stack(past_traj, dim=0)      # Shape: [batch_size, max_agents, obs_len, 2]
    future_traj = torch.stack(future_traj, dim=0)  # Shape: [batch_size, max_agents, pred_len, 2]

    return {'past_traj': past_traj, 'future_traj': future_traj}

def anorm(p1, p2):
    NORM = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    if NORM == 0:
        return 0
    return 1 / NORM

def loc_pos(seq_):
    # seq_ [obs_len, N, 2]
    obs_len = seq_.shape[0]
    num_ped = seq_.shape[1]

    pos_seq = np.arange(1, obs_len + 1)
    pos_seq = pos_seq[:, np.newaxis, np.newaxis]
    pos_seq = pos_seq.repeat(num_ped, axis=1)

    result = np.concatenate((pos_seq, seq_), axis=-1)

    return result

def seq_to_graph(seq_, seq_rel, pos_enc=False):
    # seq_ = seq_.squeeze()
    # seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    V = np.zeros((seq_len, max_nodes, 2))
    for s in range(seq_len):
        step_ = seq_[:, :, s]
        step_rel = seq_rel[:, :, s]
        for h in range(len(step_)):
            V[s, h, :] = step_rel[h]

    if pos_enc:
        V = loc_pos(V)

    return torch.from_numpy(V).type(torch.float)

def poly_fit(traj, traj_len, threshold):
    """
    Determine if a trajectory is linear or non-linear based on polynomial fitting.

    Args:
        traj (numpy.ndarray): Trajectory data of shape (2, traj_len).
        traj_len (int): Length of the trajectory.
        threshold (float): Threshold to decide linearity.

    Returns:
        float: 1.0 for non-linear, 0.0 for linear trajectories.
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)

class TrajectoryDataset(Dataset):
    """Dataloader for the Trajectory datasets"""

    def __init__(
            self, args, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
            min_ped=1, delim='\t', return_index=False, max_agents=60, noise_var=0.0):
        """
        Args:
            noise_var (float): Variance scaling factor for additive Gaussian noise.
        """
        super(TrajectoryDataset, self).__init__()

        self.return_index = return_index
        self.args = args
        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.max_agents = max_agents
        self.scale = args.scale if hasattr(args, 'scale') else 1.0
        self.noise_var = noise_var  # New noise parameter

        # [Original data loading code remains unchanged...]
        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files if _path.endswith('.txt')]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        start_frame_id_list = []

        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame, len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                _start_frame_ids = []
                for _, ped_id in enumerate(peds_in_curr_seq):
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] == ped_id, :]
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                    start_frame_id = curr_ped_seq[0, 0]
                    end_frame_id = curr_ped_seq[-1, 0]
                    if pad_end - pad_front != self.seq_len:
                        continue
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    _non_linear_ped.append(poly_fit(curr_ped_seq, self.pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                    _start_frame_ids.append(start_frame_id)

                if num_peds_considered >= min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    start_frame_id_list.append(np.unique(_start_frame_ids) * np.ones(num_peds_considered))

        if self.max_agents < self.max_peds_in_frame:
            print(f"[WARNING] max_agents ({self.max_agents}) < max agents in frame ({self.max_peds_in_frame})")

        self.start_frame_id_list = np.concatenate(start_frame_id_list, axis=0)
        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)

        # Convert to tensors
        self.obs_traj = torch.from_numpy(seq_list[:, :, :self.obs_len]).float()
        self.pred_traj = torch.from_numpy(seq_list[:, :, self.obs_len:]).float()
        self.obs_traj_rel = torch.from_numpy(seq_list_rel[:, :, :self.obs_len]).float()
        self.pred_traj_rel = torch.from_numpy(seq_list_rel[:, :, self.obs_len:]).float()
        self.loss_mask = torch.from_numpy(loss_mask_list).float()
        self.non_linear_ped = torch.from_numpy(non_linear_ped).float()
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]

        # Convert to Graphs (unchanged)
        self.v_obs = []
        self.v_pred = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end))
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_ = seq_to_graph(self.obs_traj[start:end, :], self.obs_traj_rel[start:end, :], True)
            self.v_obs.append(v_.clone())
            v_ = seq_to_graph(self.pred_traj[start:end, :], self.pred_traj_rel[start:end, :], False)
            self.v_pred.append(v_.clone())
        pbar.close()

        # Compute data variance for noise
        self.data_var = self._compute_data_variance()
    def __len__(self):
        return self.num_seq  # Already exists in your code
    def _compute_data_variance(self):
        """Compute variance for each position (agent, timestep, x/y) across all sequences."""
        all_past_trajs = []
        for idx in range(len(self.seq_start_end)):
            start, end = self.seq_start_end[idx]
            # Get original trajectory data (not scaled)
            past_traj = self.obs_traj[start:end, :]  # Shape: [num_agents, 2, obs_len]
            past_traj = past_traj.permute(0, 2, 1)  # [num_agents, obs_len, 2]
            # Scale and pad
            past_traj_scaled = past_traj * self.scale
            num_agents = past_traj_scaled.shape[0]
            if num_agents < self.max_agents:
                pad_size = self.max_agents - num_agents
                pad_tensor = torch.zeros((pad_size, self.obs_len, 2))
                past_traj_padded = torch.cat([past_traj_scaled, pad_tensor], dim=0)
            else:
                past_traj_padded = past_traj_scaled[:self.max_agents]
            all_past_trajs.append(past_traj_padded)
        
        # Stack all sequences and compute variance
        all_past_trajs = torch.stack(all_past_trajs, dim=0)  # [num_seq, max_agents, obs_len, 2]
        var_per_index = torch.var(all_past_trajs, dim=0)     # [max_agents, obs_len, 2]
        return var_per_index

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        # Scale trajectories during retrieval
        past_traj = self.obs_traj[start:end, :].permute(0, 2, 1) * self.scale  # [num_agents, obs_len, 2]
        future_traj = self.pred_traj[start:end, :].permute(0, 2, 1) * self.scale

        # Pad/truncate to max_agents
        num_agents = past_traj.shape[0]
        if num_agents < self.max_agents:
            pad_size = self.max_agents - num_agents
            past_traj = torch.cat([past_traj, torch.zeros(pad_size, self.obs_len, 2)], dim=0)
            future_traj = torch.cat([future_traj, torch.zeros(pad_size, self.pred_len, 2)], dim=0)
        elif num_agents > self.max_agents:
            past_traj = past_traj[:self.max_agents]
            future_traj = future_traj[:self.max_agents]

        # Add Gaussian noise to past trajectory
        if self.noise_var > 0:
            std = torch.sqrt(self.noise_var * self.data_var)  # [max_agents, obs_len, 2]
            noise = torch.randn_like(past_traj) * std
            past_traj += noise

        if self.return_index:
            frame_id = self.start_frame_id_list[start:end][0]
            return {'past_traj': past_traj, 'future_traj': future_traj, 'frame_id': frame_id}
        return {'past_traj': past_traj, 'future_traj': future_traj}