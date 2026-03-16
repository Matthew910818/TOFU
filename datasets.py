import os
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional, Tuple
from tqdm import tqdm
from scipy.signal import find_peaks
from utils import (
    to_str_array, 
    reshape_ev_to_grid, 
    reshape_dxdy_to_grid,
    mad_stats, 
    robust_zscore, 
    moving_average_1d, 
    hist_entropy_1d, 
    hist_entropy_2d
)

# -----------------------------
# Data Processing Helpers
# -----------------------------
def build_x(ev_w: np.ndarray, dxdy_w: np.ndarray) -> torch.Tensor:
    """Build the model input tensor using temporal differences (Velocity) instead of absolute positions."""
    ev_w = np.log1p(ev_w).astype(np.float32)
    delta_dxdy = np.zeros_like(dxdy_w)
    if dxdy_w.shape[0] > 1:
        delta_dxdy[1:] = dxdy_w[1:] - dxdy_w[:-1]
    delta_dx = delta_dxdy[:, :, :, 0].astype(np.float32)
    delta_dy = delta_dxdy[:, :, :, 1].astype(np.float32)
    x = np.stack([ev_w, delta_dx, delta_dy], axis=1)  
    return torch.from_numpy(x)

def read_trial_arrays(path: str, H: int, W: int):
    """Read .npz file and return reshaped arrays."""
    npz = np.load(path, allow_pickle=True)
    ev_key = "ev"
    
    ev = reshape_ev_to_grid(npz[ev_key], H, W).astype(np.float32)
    dxdy = reshape_dxdy_to_grid(npz["dxdy"], H, W).astype(np.float32)
    phase = to_str_array(npz["phase"]) if "phase" in npz.files else None
    compliance = None
    if "compliance" in npz.files:
        compliance = str(npz["compliance"]).strip()
    return ev, dxdy, phase, compliance

def compute_frame_features(ev: np.ndarray, dxdy: np.ndarray):
    """Compute basic spatial features for each frame in the window."""
    dx = dxdy[..., 0]
    dy = dxdy[..., 1]
    disp_mag = np.sqrt(dx * dx + dy * dy)
    event_sum = ev.reshape(ev.shape[0], -1).sum(axis=1)
    disp_mean = disp_mag.reshape(disp_mag.shape[0], -1).mean(axis=1)
    dx_mean = dx.reshape(dx.shape[0], -1).mean(axis=1)
    dy_mean = dy.reshape(dy.shape[0], -1).mean(axis=1)

    d_disp = np.zeros_like(disp_mean)
    if disp_mean.size > 1:
        d_disp[1:] = np.abs(np.diff(disp_mean))

    d_event = np.zeros_like(event_sum)
    if event_sum.size > 1:
        d_event[1:] = np.abs(np.diff(event_sum))

    return {
        "event_sum": event_sum.astype(np.float32),
        "disp_mean": disp_mean.astype(np.float32),
        "dx_mean": dx_mean.astype(np.float32),
        "dy_mean": dy_mean.astype(np.float32),
        "d_disp": d_disp.astype(np.float32),
        "d_event": d_event.astype(np.float32),
    }

def compute_peak_score(ev: np.ndarray, dxdy: np.ndarray, smooth_k: int = 5):
    """
    Build a handcrafted frame score for compliance sampling.
    Emphasizes high event response, large displacement, and strong changes.
    """
    feat = compute_frame_features(ev, dxdy)
    event_z = robust_zscore(moving_average_1d(feat["event_sum"], smooth_k))
    disp_z = robust_zscore(moving_average_1d(feat["disp_mean"], smooth_k))
    d_disp_z = robust_zscore(moving_average_1d(feat["d_disp"], smooth_k))
    d_event_z = robust_zscore(moving_average_1d(feat["d_event"], smooth_k))
    score = 0.35 * event_z + 0.30 * disp_z + 0.25 * d_disp_z + 0.10 * d_event_z
    return score.astype(np.float32)

def select_top_centers(score: np.ndarray, idx: np.ndarray, topk: int, min_gap: int):
    """Select top-k scoring frame indices with a minimum gap between them."""
    if idx.size == 0:
        return []
    cand = sorted([(int(t), float(score[t])) for t in idx], key=lambda x: x[1], reverse=True)
    picked: List[int] = []
    for t, _ in cand:
        if all(abs(t - p) >= min_gap for p in picked):
            picked.append(t)
        if len(picked) >= topk:
            break
    picked.sort()
    return picked

def phase_window_starts(idx: np.ndarray, win: int, stride: int):
    """Get starting indices for sliding windows within a phase."""
    if idx.size < win:
        return []
    start = int(idx[0])
    end = int(idx[-1])
    last = end - win + 1
    if last < start:
        return []
    return list(range(start, last + 1, stride))

def build_texture_stable_starts(ev: np.ndarray, dxdy: np.ndarray, idx_tex: np.ndarray, win: int, stride: int, topk_peaks: int = 5, peak_margin: int = 16, peak_smooth_k: int = 5, peak_jitter: int = 4, jitter_stride: int = 2):
    """
    Build stable baseline window starts from RECORDING_TEXTURE by excluding
    any candidate window that overlaps with peak-centered active spans.
    (Kept for backward compatibility)
    """
    if idx_tex.size < win:
        return []

    tex_start = int(idx_tex[0])
    tex_end = int(idx_tex[-1])
    last_valid = tex_end - win + 1
    if last_valid < tex_start:
        return []

    score = compute_peak_score(ev, dxdy, smooth_k=peak_smooth_k)
    centers = select_top_centers(score, idx_tex, topk=topk_peaks, min_gap=peak_margin)

    forbidden_spans = []
    offsets = list(range(-peak_jitter, peak_jitter + 1, jitter_stride))

    for c in centers:
        base_t0 = c - win // 2
        candidate_starts = []
        for off in offsets:
            t0 = int(np.clip(base_t0 + off, tex_start, last_valid))
            candidate_starts.append(t0)

        if len(candidate_starts) == 0:
            continue

        span_start = min(candidate_starts)
        span_end = max(candidate_starts) + win - 1
        forbidden_spans.append((span_start, span_end))

    if len(forbidden_spans) > 0:
        forbidden_spans.sort(key=lambda x: x[0])
        merged = [forbidden_spans[0]]
        for s, e in forbidden_spans[1:]:
            last_s, last_e = merged[-1]
            if s <= last_e + 1:
                merged[-1] = (last_s, max(last_e, e))
            else:
                merged.append((s, e))
        forbidden_spans = merged

    all_starts = list(range(tex_start, last_valid + 1, stride))

    stable_starts = []
    for t0 in all_starts:
        w_start = t0
        w_end = t0 + win - 1
        overlap = False
        for s, e in forbidden_spans:
            if not (w_end < s or w_start > e):
                overlap = True
                break
        if not overlap:
            stable_starts.append(t0)

    return stable_starts

# -----------------------------
# PyTorch Datasets
# -----------------------------
class TactileWindowDataset(Dataset):
    """
    Dataset for Pretraining.
    Loads .npz files and yields windows of shape: x (L, 3, 9, 7)
    """
    def __init__(self, data_dir: str, win: int = 16, stride: int = 4, phases: Optional[List[str]] = None, H: int = 9, W: int = 7, max_files: Optional[int] = None):
        self.data_dir = data_dir
        self.win = win
        self.stride = stride
        self.phases = phases
        self.H = H
        self.W = W

        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
        files.sort()
        if max_files is not None:
            files = files[:max_files]
        self.files = files

        self.index = []
        self._meta = []  
        for fi, path in enumerate(self.files):
            npz = np.load(path)
            ev_key = "ev"
            ev = reshape_ev_to_grid(npz[ev_key], H, W)
            T = ev.shape[0]

            if self.phases is not None and "phase" in npz.files:
                phase_arr = to_str_array(npz["phase"])
                keep = np.isin(phase_arr, np.array(self.phases, dtype=object))
                keep_idx = np.where(keep)[0]

                if keep_idx.size < self.win:
                    continue
                
                seg_starts = []
                s = 0
                while s < keep_idx.size:
                    e = s
                    while e + 1 < keep_idx.size and keep_idx[e + 1] == keep_idx[e] + 1:
                        e += 1
                    seg_len = e - s + 1
                    if seg_len >= self.win:
                        seg_starts.append((keep_idx[s], seg_len))
                    s = e + 1
                    
                for seg_start, seg_len in seg_starts:
                    for t0 in range(seg_start, seg_start + seg_len - self.win + 1, self.stride):
                        self.index.append((fi, t0))
                        self._meta.append((os.path.basename(path), t0))
            else:
                if T < self.win:
                    continue
                for t0 in range(0, T - self.win + 1, self.stride):
                    self.index.append((fi, t0))
                    self._meta.append((os.path.basename(path), t0))

        if len(self.index) == 0:
            raise RuntimeError("No windows built. Check phases / win / data shapes.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fi, t0 = self.index[idx]
        path = self.files[fi]
        npz = np.load(path)

        ev_key = "ev"
        ev = reshape_ev_to_grid(npz[ev_key], self.H, self.W)
        dxdy = reshape_dxdy_to_grid(npz["dxdy"], self.H, self.W)

        ev_w = ev[t0:t0 + self.win]
        dx_w = dxdy[t0:t0 + self.win, :, :, 0]
        dy_w = dxdy[t0:t0 + self.win, :, :, 1]

        ev_w = np.log1p(ev_w).astype(np.float32)
        dx_w = dx_w.astype(np.float32)
        dy_w = dy_w.astype(np.float32)

        x = np.stack([ev_w, dx_w, dy_w], axis=1)
        return torch.from_numpy(x)

class ComplianceDataset(Dataset):
    """
    Peak-driven compliance dataset with local sliding-window expansion.
    Labels: 0 = Soft, 1 = Hard
    """
    def __init__(self, files: List[str], win: int, stride: int, H: int = 9, W: int = 7, topk_peaks: int = 3, peak_margin: Optional[int] = None, peak_smooth_k: int = 5, peak_jitter: int = 4, jitter_stride: int = 2):
        self.items = [] 
        self.cache = {}  
        self.win = win
        self.stride = stride
        self.H = H
        self.W = W
        self.topk_peaks = topk_peaks
        self.peak_margin = win if peak_margin is None else peak_margin
        self.peak_smooth_k = peak_smooth_k
        self.peak_jitter = peak_jitter
        self.jitter_stride = jitter_stride

        for trial_id, path in enumerate(files):
            if path not in self.cache:
                self.cache[path] = read_trial_arrays(path, H, W)

            ev, dxdy, phase, compliance = self.cache[path]
            if phase is None or compliance is None:
                continue

            idx = np.where(phase == "RECORDING_TEXTURE")[0]
            if idx.size < win:
                continue

            comp = compliance.lower().strip()
            if comp not in ["soft", "hard"]:
                continue
            y = 1 if comp == "hard" else 0

            score = compute_peak_score(ev, dxdy, smooth_k=self.peak_smooth_k)
            centers = select_top_centers(score, idx, topk=self.topk_peaks, min_gap=self.peak_margin)
            
            if len(centers) == 0:
                continue

            start = int(idx[0])
            end = int(idx[-1])
            last_valid = end - win + 1
            if last_valid < start:
                continue

            offsets = list(range(-self.peak_jitter, self.peak_jitter + 1, self.jitter_stride))
            used_t0 = set()

            for c in centers:
                base_t0 = c - win // 2
                for off in offsets:
                    t0 = int(np.clip(base_t0 + off, start, last_valid))
                    if t0 in used_t0:
                        continue
                    used_t0.add(t0)
                    self.items.append((path, t0, y, trial_id))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, t0, y, trial_id = self.items[i]
        ev, dxdy, _, _ = self.cache[path]
        x = build_x(ev[t0:t0 + self.win], dxdy[t0:t0 + self.win])
        return x, torch.tensor(y, dtype=torch.long), torch.tensor(trial_id, dtype=torch.long)

class SlipOrdinalDataset(Dataset):
    """Energy-Peak Driven Slip Ordinal Dataset (Stick-slip aware)"""
    def __init__(self, files: List[str], win: int, stride: int, H: int = 9, W: int = 7, 
                 prominence: float = 0.05, slip_thresh: float = 0.40, 
                 slip_margin: int = 3, preslip_win: int = 15, **kwargs):
        self.items = []  
        self.cache = {}  
        self.win = win
        self.stride = stride
        self.H = H
        self.W = W

        # Keep dummy variables to maintain compatibility with legacy parameter calls in finetune_v2.py
        self.lo = 0.0
        self.hi = 1.0

        for path in tqdm(files, desc="Processing Trials", leave=False):
            if path not in self.cache:
                self.cache[path] = read_trial_arrays(path, H, W)

            ev, dxdy, phase, comp = self.cache[path]
            if phase is None:
                continue

            idx_slp = np.where(phase == "COLLECTING_SLIP")[0]
            if len(idx_slp) < win:
                continue

            # Extract data specifically for the slip phase
            ev_slp = ev[idx_slp]
            dxdy_slp = dxdy[idx_slp]
            L = len(ev_slp)

            # 1. Calculate variations: Velocity and Event Sum
            delta_dxdy = np.zeros_like(dxdy_slp)
            if L > 1:
                delta_dxdy[1:] = dxdy_slp[1:] - dxdy_slp[:-1]
            
            # Use reshape to avoid tuple axis errors
            v_mag = np.sqrt(delta_dxdy[..., 0]**2 + delta_dxdy[..., 1]**2).reshape(L, -1).mean(axis=1)
            e_sum = ev_slp.reshape(L, -1).sum(axis=1)

            # 2. Smoothing and Global Normalization (Local Min-Max Norm)
            v_smooth = moving_average_1d(v_mag, k=5)
            e_smooth = moving_average_1d(e_sum, k=5)

            v_min, v_max = v_smooth.min(), v_smooth.max()
            e_min, e_max = e_smooth.min(), e_smooth.max()
            v_norm = (v_smooth - v_min) / (v_max - v_min + 1e-8)
            e_norm = (e_smooth - e_min) / (e_max - e_min + 1e-8)

            # Comprehensive energy E(t), aligned so the peak is exactly 1.0
            E = v_norm + e_norm
            E_min, E_max = E.min(), E.max()
            E = (E - E_min) / (E_max - E_min + 1e-8)

            # 3. Peak Detection
            peaks, _ = find_peaks(E, prominence=prominence, distance=10)

            # Crop the excessively long, meaningless dead water at the tail 
            # (keep 200 frames after the last detected action)
            if len(peaks) > 0:
                last_action = peaks[-1]
            else:
                last_action = np.argmax(E)
            valid_end = min(L, last_action + 200)

            # 4. Time-window Backcasting for labeling
            labels = np.zeros(L, dtype=int)
            for p in peaks:
                peak_energy = E[p]
                
                # Case A: Huge Peak -> Label as Class 2 (Slip)
                if peak_energy >= slip_thresh:
                    s_start = max(0, p - slip_margin)
                    s_end = min(L, p + slip_margin + 2)
                    labels[s_start:s_end] = 2
                    
                    # The rising slope before a major slip is labeled as Class 1 (Pre-slip)
                    pre_start = max(0, p - preslip_win)
                    for i in range(pre_start, s_start):
                        if labels[i] == 0:
                            labels[i] = 1
                            
                # Case B: Small Peak (micro-slip/deformation) -> Downgrade to Class 1 (Pre-slip)
                else:
                    s_start = max(0, p - slip_margin)
                    s_end = min(L, p + slip_margin + 2)
                    for i in range(s_start, s_end):
                        if labels[i] == 0:
                            labels[i] = 1

            # 5. Generate Sliding Windows
            # Only sample up to valid_end to avoid collecting too many meaningless Class 0 (Stable) windows
            for t_local in range(0, valid_end - win + 1, stride):
                # The prediction target for the window is the physical state of its *last frame*
                y = labels[t_local + win - 1]
                
                # Record the absolute position idx_slp[t_local] for __getitem__ retrieval
                self.items.append((path, int(idx_slp[t_local]), int(y)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, t0, y = self.items[i]
        ev, dxdy, _, _ = self.cache[path]
        
        # Call build_x (which has already been updated to compute temporal differences)
        x = build_x(ev[t0:t0 + self.win], dxdy[t0:t0 + self.win])
        
        return x, torch.tensor(y, dtype=torch.long)