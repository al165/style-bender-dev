import sys
from itertools import groupby

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import interp1d

import torch
from torchnmf.nmf import NMFD

import librosa
import librosa.display

sys.path.append("../COMMON_UTILS/")
from utils import env_sample, normalise


def getDecomposition(
    y: np.array,
    window_size: int = 2048,
    T: int = 10,
    R: int = 1,
    H: np.ndarray = None,
    W: np.ndarray = None,
    beta: float = 1,
    max_iter: int = 1000,
    tol: float = 0.0001,
    l1_ratio: float = 1,
    sW=None,
    sH=None,
    sparse_fit: bool = False,
    device="cpu",
    trainable_W: bool = True,
    trainable_H: bool = True,
    W_mask: np.ndarray = None,
    H_mask: np.ndarray = None,
    hop_length: int = 512,
):
    """Decompose drum audio `y`.

    Returns W, H, V, phi, net."""

    try:
        y = torch.from_numpy(y)
    except TypeError:
        pass

    S = torch.stft(
        y, window_size, window=torch.hann_window(window_size), return_complex=True, hop_length=hop_length,
    ).to(device)

    A = torch.abs(S).unsqueeze(0)
    phi = torch.angle(S).cpu().detach().numpy()

    net = NMFD(
        A.shape, 
        H=H, 
        W=W, 
        rank=R, 
        T=T, 
        trainable_W=trainable_W, 
        trainable_H=trainable_H
    ).to(device)

    # probably need to decouple...
    net.phi = phi
    net.S = S
        
    # bug...
    if W is not None:
        net.W = torch.nn.Parameter(W.detach().clone().to(device), requires_grad=trainable_W)
    if H is not None:
        net.H = torch.nn.Parameter(H.detach().clone().to(device), requires_grad=trainable_H)
        
    if W_mask is not None and trainable_W:
        net.W.register_hook(lambda grad: grad * W_mask.float())   
    if H_mask is not None and trainable_H:
        net.H.register_hook(lambda grad: grad * H_mask.float())

    if not sparse_fit:
        net.fit(A, beta=beta, max_iter=max_iter, tol=tol, l1_ratio=l1_ratio)
    else:
        net.sparse_fit(A, sW=sW, sH=sH, beta=beta, max_iter=max_iter)

    W = net.W.detach().cpu().numpy()
    H = net.H.squeeze(0).detach().cpu().numpy()
    V = net().squeeze().detach().cpu().numpy()

    return W, H, V, phi, net


def plotDecomposition(W, H, V, sr: int, S=None, hop_length: int = 512):
    """Plot decomposition.

    Parameters
    ----------
    W : np.array
    H : np.array
    V : np.array

    """
    R = W.shape[1]
    fig, axes = plt.subplots(
        R + 1, 2, gridspec_kw={"width_ratios": [1, 4]}, figsize=(15, R * 3)
    )
    fig.delaxes(axes[-1, 0])

    for i in range(R):
        librosa.display.specshow(
            librosa.amplitude_to_db(W[:, i], ref=np.max), 
            y_axis="log", 
            ax=axes[i, 0],
            sr=sr,
            hop_length=hop_length,
        )
        axes[i, 0].set_title(f"W_{i}")
        axes[i, 1].plot(H[i])
        axes[i, 1].set_title(f"H_{i}")
        axes[i, 1].set_xlim(0, H.shape[1])
        axes[i, 1].set_xticks([])

    librosa.display.specshow(
        librosa.amplitude_to_db(V, ref=np.max),
        y_axis="log",
        x_axis="time",
        ax=axes[-1, 1],
        sr=sr,
        hop_length=hop_length,
    )
    axes[-1, 1].set_title("V = HW")
    plt.show()


def isolateSources(net, phi=None, device="cpu", hop_length: int = 512):
    """
    Parameters
    ----------
    net : NMFD model
    phi : np.array, optional
    """

    ys = []

    for c in range(net.H.shape[1]):
        H_c = torch.zeros_like(net.H).to(device)
        H_c[:, c, :] = net.H[:, c, :]

        V_hat = net.reconstruct(net.W, H_c).cpu().squeeze().detach().numpy()

        if phi is not None:
            S_r = V_hat * np.exp(1j * phi)
            y_r = librosa.istft(S_r, hop_length=512)
        else:
            # use Griffin-Lim algorithm for phase recovery
            y_r = librosa.griffinlim(V_hat)
        ys.append(y_r)

    return ys


def drumGridFromHs(Hs: list, grid_size: int = 48) -> np.array:
    """
    Parameters
    ----------
    Hs : list[np.array]
        List of activations H of shape (number_ins, time)
    grid_size : int
        Size of drum grid. Defualt 48

    Returns
    -------
    grids : np.array
    """

    grids = np.zeros((len(Hs), len(Hs[0]), grid_size))

    for i, H in enumerate(Hs):
        for j, act in enumerate(H):
            if len(act) == 0:
                continue
            f = interp1d(np.linspace(0, grid_size + 1, len(act)), act)
            reduced = f(np.arange(grid_size))

            peaks, _ = find_peaks(np.concatenate([[0], reduced]), prominence=0.05)
            peaks -= 1

            grids[i, j, peaks] = reduced[peaks]

    # normalise over each track
    for i in range(len(Hs[0])):
        grids[:, i, :] /= grids[:, i, :].max()

    return grids

def quantiseGrids(grids: list) -> np.array:
    '''Adjustment to alter grid to most likly configuration. 
    May carry over to next grid!
    '''
    
    if len(grids.shape) == 2:
        grids = grids.reshape((1, *grids.shape))
    
    n, ins, gs = grids.shape
    
    grids_linear = np.concatenate(grids, axis=1)
    
    prior = makeGridPrior(gs)
    map_ = getPriorMap(prior)
    
    q_grid = np.zeros_like(grids_linear)
    q_grid = np.concatenate([q_grid, np.zeros((ins, 1))], axis=1)
    
    for k in range(n):
        for j in range(ins):
            for i in range(gs):
                d = map_[i]
                cell = k*gs + i
                try:
                    q_grid[j, cell+d] = max(grids_linear[j, cell], q_grid[j, cell+d])
                except IndexError:
                    print(f'{cell=} {d=} {j=} ({k=} * {n=} + {i=} = {cell})')
    
    q_grid = q_grid[:, :-1]
    q_grid = np.split(q_grid, n, axis=1)
    q_grid = np.stack(q_grid)
        
    return q_grid

def makeGridPrior(grid_size: int = 48, alpha: float = 0.5) -> np.array:

    assert 0.0 <= alpha <= 1.0
    
    weights = np.zeros(grid_size)
    i = 2
    while True:
        skip = int(grid_size/(2**i))
        if skip == 0:
            break
        weights[::skip] += 1
        i += 1
    
#     i = 2
#     while True:
#         skip = int(grid_size/(3* (2**i)))
#         if skip == 0:
#             break
#         weights[::skip] += 0.5
#         i += 1
    
    # scale so min value is alpha
    weights = weights * (1 - alpha) + alpha
    
    return weights / weights.max()

def getPriorMap(p: np.array) -> np.array:
    map_ = np.zeros(len(p), dtype=int)
    p_ = np.zeros(len(p)+2)
    p_[1:-1] = p
    p_[-1] = p[0]
    p_[0] = p[-1]
#     p_[-1] = np.inf
    
    for i in range(1, len(p)+1):
        if p_[i-1] > p_[i+1] and p_[i] < p_[i-1]:
            map_[i-1] = -1
        elif p_[i-1] < p_[i+1] and p_[i] < p_[i+1]:
            map_[i-1] = 1
            
    return map_

def splitActivationsAndAudio(
    ys: np.array, 
    H: np.array, 
    downbeats: np.array, 
    sr: int = 22050, 
    delay: int = 1, 
    hop_length: int = 512,
) -> tuple:

    """
    Parameters
    ----------
    y : np.array
    H : np.array
    downbeats : np.array
    sr : int
        Default 22050.
    delay : int
        Delay (in frames) to shift H to align activations with beat grid. Default 1.

    Returns
    -------
    H_bars : list[np.array]
    bars : list[np.array]
    """

    bar_frames = librosa.time_to_frames(downbeats, sr=sr, hop_length=hop_length) - 1
    bar_samples = librosa.time_to_samples(downbeats, sr=sr)
    y_sep = np.stack(ys)
    H_bars = []
    bars = []
    for beat in range(len(downbeats) - 1):
        H_bars.append(np.copy(H[:, bar_frames[beat] : bar_frames[beat + 1]]))
        bars.append(np.copy(y_sep[:, bar_samples[beat] : bar_samples[beat + 1]]))

    return H_bars, bars


def separateSamples(y: np.array, sr: int, onsets: np.array, hop_length: int = 512) -> list:
    if len(onsets) <= 1:
        return []

    samples = []
    frames = librosa.frames_to_samples(onsets, hop_length=hop_length)
    for i in range(len(onsets) - 1):
        sample = y[frames[i] : frames[i + 1]]
        sample = trimSample(sample)
        if len(sample) < librosa.time_to_samples(0.1, sr=sr):
            continue
        sample = env_sample(sample)
        samples.append(sample)

    return samples


def longestRun(arr: np.array) -> tuple:
    """Returns the start index and length of longest run of consecutive Trues."""
    idx = 0
    max_len = 0
    max_idx = 0
    for k, g in groupby(arr):
        g_len = len(list(g))
        if k and g_len > max_len:
            max_len = g_len
            max_idx = idx

        idx += g_len

    return max_idx, max_len


def trimSample(y: np.array, padding: int = 5, hop_length: int = 512) -> np.array:
    rms = np.mean(librosa.feature.rms(y=y, hop_length=hop_length), axis=0)
    sound = rms > 0.001

    start_frame, max_len = longestRun(sound)
    end_frame = max_len + padding

    start_idx, end_idx = librosa.frames_to_samples([start_frame, end_frame], hop_length=hop_length)

    return y[start_idx:end_idx]


def isolatePeak(H: np.array, index: int):
    """Returns tuple (start, end)"""
    start = index
    h = H[start]
    while start > 0 and h >= H[start] and H[start] > 0:
        h = H[start]
        start -= 1

    end = index
    h = H[end]
    while end < len(H) and h >= H[end] and H[end] > 0:
        h = H[end]
        end += 1

    return start, end


def getSamples(net: NMFD, chan: int, sr: int, hop_length: int = 512) -> list:

    H = net.H[0, chan].cpu().detach().numpy()

    # peaks from activations H
    peaks, _ = find_peaks(H, prominence=5)
    # subsample them
    peaks = peaks[1::3]

    # peaks and zero their surroundings
    iso_peaks = []
    for peak in peaks:
        peak_range = isolatePeak(H, peak)
        iso_peaks.extend(list(range(*peak_range)))
    mask = np.ones(len(H), bool)
    mask[iso_peaks] = 0
    H[mask] = 0

    # reconstruct audio with isolated samples
    V_clean = (
        net.forward(
            torch.tensor(H).unsqueeze(0).unsqueeze(0).to(net.W.device),
            net.W[:, chan, :].unsqueeze(1),
        )
        .cpu()
        .squeeze()
        .detach()
        .numpy()
    )

    S_clean = V_clean * np.exp(1j * net.phi)
    y_clean = librosa.istft(S_clean, hop_length=hop_length)

    # use onsets of activations to indicate sample start times
    onsets = librosa.onset.onset_detect(onset_envelope=H, backtrack=True, wait=0, sr=sr, hop_length=hop_length)
    samples = separateSamples(y_clean, sr, onsets, hop_length=hop_length)

    return samples


def reconstructDrums(H: np.ndarray, samples: list, length: int, hop_length: int = 512) -> np.ndarray:
    if len(H) != len(samples):
        raise ValueError(
            f'H shape ({H.shape}) does not match number of samples ({len(samples)})'
        )
    
    y_rec = np.zeros(length)
    
    for j, sample in enumerate(samples):
        act = H[j]
        peaks, _ = find_peaks(np.insert(act, 0, 0), prominence=3)
        peaks -= 1
        time = librosa.frames_to_samples(peaks, hop_length=hop_length)
        
        for i, p in enumerate(peaks):
            amp = act[p]# / act.max()
            sl = min(len(sample), length - time[i])
            
            y_rec[time[i]:time[i]+sl] += amp*sample[:sl]
            
    y_rec = normalise(y_rec)
            
    return y_rec
