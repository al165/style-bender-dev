import os
import subprocess

import numpy as np

import librosa
import soundfile as sf

import torch

from IPython.display import display, Audio


def run_cmd(cmd: list):
    p = subprocess.Popen(cmd)
    p.wait()
    if p.returncode != 0:
        print("command failed")

    return p.returncode


def get_audio_files(folder, extensions=["mp3", "wav", "flac", "ogg"]):
    all_files = []
    for root, _, files in os.walk(folder, topdown=True):
        for f in files:
            if f.split(".")[-1] in extensions:
                all_files.append(str(os.path.join(root, f)))

    return all_files


def save_audio(y: np.array, sr: int, fn: str) -> None:
    """Save audio to file"""
    sf.write(fn, y, sr)


def hash_audio(y: np.array) -> int:
    h = str(hash(y.data.tobytes()))
    return h


def play(y: np.array, sr: int, autoplay: bool = False, normalize: bool = False):
    display(Audio(y, rate=sr, autoplay=autoplay, normalize=normalize))


def plot_beat_grid(trig: np.array, ax=None):
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

    ax.matshow(trig, cmap="gray_r")
    ax.set_xticks([0, 12, 24, 36])

    return ax


def plot_audio(y: np.ndarray, sr: int = 44100, ax=None, **kwargs):
    if ax is None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()

    t = librosa.samples_to_time(np.arange(len(y)), sr=sr)
    ax.plot(t, y, **kwargs)

    return ax


def normalise(y: np.ndarray) -> np.ndarray:
    if y.max() > 1.0 or y.min() < -1.0:
        y /= max(y.max(), -y.min())
    return y


def split_audio_bars(y: np.array, downbeats: np.array, sr: int) -> list:
    import librosa

    bar_idxs = librosa.time_to_samples(downbeats, sr)
    bars = []
    for i in range(len(bar_idxs) - 1):
        bars.append(y[bar_idxs[i] : bar_idxs[i + 1]])

    return bars


def make_click_track(
    trig: np.array,
    length: int,
    sr: int = 22050,
    freq: float = 300,
    duration: float = 0.1,
) -> np.array:
    import librosa

    if len(trig.shape) == 1:
        trig = np.array([trig])

    y_final = np.zeros(length)
    step = (length / sr) / trig.shape[1]

    try:
        _ = iter(freq)
    except TypeError:
        freq = np.ones(trig.shape[0]) * freq
    else:
        if len(freq) < trig.shape[0]:
            freq = np.ones(trig.shape[0]) * freq[0]

    try:
        _ = iter(duration)
    except TypeError:
        duration = np.ones(trig.shape[0]) * duration
    else:
        if len(duration) < trig.shape[0]:
            duration = np.ones(trig.shape[0]) * duration[0]

    for i, row in enumerate(trig):
        times = np.where(row > 0.9)[0] * step
        y_clicks = librosa.clicks(
            times=times, sr=sr, click_freq=freq[i], click_duration=duration[i]
        )

        if len(y_clicks) > len(y_final):
            y_clicks = y_clicks[: len(y_final)]

        y_final[: len(y_clicks)] += y_clicks

    return y_final


def env_sample(y: np.array, length: int = 128, power: float = 2) -> np.array:
    """Apply amplitude envelope to the start and end of an audio sample."""

    length = abs(length)

    y_new = np.copy(y)

    env = np.linspace(0, 1, length)
    env = np.power(env, power)

    y_new[:length] *= env
    y_new[-length:] *= np.flip(env)

    return y_new


def mix_samples(
    y: np.array,
    grid: np.array,
    fps: list,
    step: float,
    sr: int = 44100,
    verbose: bool = False,
) -> np.array:

    print("/!\\ DEPRECIATED")
    y_new = np.array(y)

    """Mix in samples to an audio array `y`."""

    idxs = np.where(grid > 0.5)[0]
    if verbose:
        print(idxs)
        print(fps)

    if len(idxs) != len(fps):
        raise ValueError(f"Mismatch lengths -- idxs: {len(idxs)}, fns: {len(fps)}")

    for i, t in enumerate(idxs):
        y_sample, _ = librosa.load(fps[i], sr=sr)
        start = int(t * step)

        # trim sample if too long
        max_length = len(y_new) - start
        if verbose:
            print(
                f"mixing {fps[i]} at index {start}, (length {len(y_sample)}, max length {max_length})"
            )
        y_sample = y_sample[:max_length]
        y_sample = env_sample(y_sample, length=256)

        y_new[start : start + len(y_sample)] += y_sample

    return y_new


def mix_sample(
    y_track: np.array,
    y_sample: np.array,
    where: int,
    amp: float = 1.0,
    env: bool = False,
) -> np.array:
    """
    Parameters
    ----------
    y_track : np.array
        audio signal to add sample to.
    y_sample : np.array
        audio signal of sample.
    where : int
        index of start of audio signal.
    amp : float
        amp modulation. Defualt 1.0.
    env : bool (optional)
        should envelope sample. Default False.

    Returns
    -------
    y_track : np.array
    """

    length = min(len(y_track) - where, len(y_sample))
    if env:
        y_sample = env_sample(y_sample)

    y_track[where : where + length] += y_sample[:length] * amp

    return y_track


def create_path(start, end, n):
    path = np.zeros((n, len(start)))
    for i in range(len(start)):
        path[:, i] = np.linspace(start[i], end[i], n)
    return path


def create_point_path(
    *points,
    n: int = 4,
    pause: int = 1,
    device: str = "cpu",
    include_first=True,
) -> torch.Tensor:
    """
    Parameters
    ----------
    *points : list[torch.Tensor]
    n : int
        Number of steps between each point
    pause : int
        Number of times to repeat point

    Returns
    -------
    path : np.array
    """

    if len(points) == 0:
        raise ValueError("did not receive any points")

    path = []

    for i, point in enumerate(points[:-1]):
        if i == 0 and include_first:
            path.extend([point] * pause)

        segment = torch.zeros((n, len(point)), device=device)
        for j in range(len(point)):
            # torch.linspace does not implement `endpoint` parameter...
            segment[:, j] = torch.linspace(point[j], points[i + 1][j], n + 1)[:-1].to(
                device
            )

        path.extend(list([x for x in segment]))

    path.extend([points[-1]] * pause)

    path = torch.stack(path)

    return path
