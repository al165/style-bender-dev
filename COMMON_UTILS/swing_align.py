import os
import pickle as pkl

import numpy as np
from scipy.signal import find_peaks

import librosa

from drum_processor import getDownbeats
from tempo_align import matchAudioEvents


def divideTimes(db: np.ndarray, n: int = 4) -> np.ndarray:
    """
    Parameters
    ----------
    db : np.ndarray
    n : int
        Default 4.

    Returns
    -------
    beats : np.ndarray

    """
    times = []
    for i in range(len(db) - 1):
        times.extend(np.linspace(db[i], db[i + 1], n, endpoint=False))
    times = np.array(times)

    return times


def getSwingPoints(
    y: np.ndarray, sr: int, db: np.ndarray, hop_length: int = 256
) -> np.ndarray:
    """
    Parameters
    ----------
    y : np.ndarray
    sr : int
    db : np.ndarray
    hop_length : int
        Default 256

    Returns
    -------
    points : np.ndarray

    """
    beats = divideTimes(db)

    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    beats_rms = []
    beats_frames = librosa.time_to_frames(beats, sr=sr, hop_length=hop_length)
    for start, end in zip(beats_frames[:-1], beats_frames[1:]):
        beats_rms.append(rms[start:end])

    length = min(map(len, beats_rms))
    beat_energy = np.mean(np.stack([b[:length] for b in beats_rms]), axis=0)

    peaks, _ = find_peaks(beat_energy, height=0.03, prominence=0.005)

    points = []
    for p in peaks:
        swing = (p - peaks[0]) / length
        if abs(swing - 0.5) < 0.05:
            swing = 0.5

        swing = round(swing, 2)
        points.append(swing)

    if len(points) == 1:
        points.append(0.5)

    return np.array(points)


def getSwingMap(a: np.ndarray, b: np.ndarray) -> list:
    """
    Returns the list of points (x, y) where x is mapped to position y.
    """
    a = np.array(a)
    b = np.array(b)
    map_ = []
    if len(a) == len(b):
        for x, y in zip(a, b):
            map_.append((x, y))
    elif len(a) < len(b):
        for x in a:
            y = b[np.argmin(np.abs(b - x))]
            map_.append((x, y))
    else:
        for y in b:
            x = a[np.argmin(np.abs(a - y))]
            map_.append((x, y))

    return map_


def plotSwingMap(a: np.ndarray, b: np.ndarray, map_: list = None, ax=None):
    """
    Parameters
    ----------
    a : np.ndarray
    b : np.ndarray
    map : list (optional)
    ax : Axes (optional)

    Returns
    -------
    ax : Axes
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 1))

    ax.plot(a, np.zeros_like(a) + 1, marker="o", ls="", ms=15, c="k")
    ax.plot(b, np.zeros_like(b), marker="o", ls="", ms=15, c="k")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["to", "from"])
    if map_ is not None:
        for m in map_:
            ax.arrow(m[0], 1, m[1] - m[0], -1, color="red", ls=":", lw=1)

    return ax


def getSwingTimings(db: np.ndarray, map_: list) -> tuple:
    """
    Parameters
    ----------
    db : np.ndarray
    map_ : list[tuple]

    Returns
    -------
    points_from : np.ndarray
    points_to : np.ndarray
    """
    beats = divideTimes(db)

    points_from = []
    points_to = []

    for i in range(len(beats) - 1):
        dt = beats[i + 1] - beats[i]
        for m in map_:
            points_from.append(beats[i] + dt * m[0])
            points_to.append(beats[i] + dt * m[1])

    return points_from, points_to


def alignSwing(
    y_org: np.ndarray,
    y_trg: np.ndarray,
    sr: int,
    db_org: np.ndarray,
    db_trg: np.ndarray,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Computes the swing/groove of `y_trg` and applies it to `y_org`.

    Parameters
    ----------
    y_org : np.ndarray
    y_trg : np.ndarray
    sr : int
    db_org : np.ndarray
    db_trg : np.ndarray
    hop_length : int (optional)
        Default 256.

    Returns
    -------
    y_warped : np.ndarray
        Warped audio signal.
    """

    points_org = getSwingPoints(y_org, sr, db_org, hop_length=hop_length)
    points_trg = getSwingPoints(y_trg, sr, db_trg, hop_length=hop_length)

    map_ = getSwingMap(points_org, points_trg)
    points_from, points_to = getSwingTimings(db_org, map_)

    y_warped = matchAudioEvents(y_org, sr, points_from, points_to, hq=True)
    return y_warped
