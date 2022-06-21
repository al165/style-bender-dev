import warnings

import numpy as np
from scipy.interpolate import interp1d

import librosa


def quantiseAudio(y: np.ndarray, sr: int, db: np.array, hq=False) -> np.ndarray:
    """
    Warps audio such that the given downbeats `db` are regularly spaced
    """

    if hq:
        import pyrubberband

    if len(db) < 3:
        warnings.warn(
            f"Not enough downbeats (len(db)={len(db)}), returning original audio"
        )
        return y

    db_q = np.linspace(db[0], db[-1], len(db))
    db_samples = librosa.time_to_samples(db, sr=sr)
    q_bar_len = librosa.time_to_samples(db_q[1] - db_q[0], sr=sr)

    # split audio into bars
    bars = np.split(y, db_samples)

    y_q = [bars[0]]
    for i, b in enumerate(bars[1:-1]):
        fix_rate = len(b) / q_bar_len
        if hq:
            b_q = pyrubberband.pyrb.time_stretch(y=b, sr=sr, rate=fix_rate)
        else:
            b_q = librosa.effects.time_stretch(y=b, rate=fix_rate)
        y_q.append(b_q)
    y_q.append(bars[-1])

    return np.concatenate(y_q), db_q


def warpAudio(
    y_src: np.ndarray,
    y_dst: np.ndarray,
    db_src: np.ndarray,
    db_dst: np.ndarray,
    sr: int,
) -> tuple:

    """
    Warps the `src` audio to match the downbeats of `dst`.

    Parameters
    ----------
    y_src : np.ndarray
    y_dst : np.ndarray
    db_src : np.ndarray
    db_dst : np.ndarray
    sr : int

    Returns
    -------
    y_src_warped : np.ndarray
    y_dst_synced : np.ndarray
    downbeats : np.ndarray
    idxs_src : tuple
    idxs_dst : tuple
    """

    num_bars = min(len(db_dst), len(db_src))

    idxs_src = librosa.time_to_samples((db_src[0], db_src[num_bars - 1]), sr=sr)

    mapping = interp1d(
        x=np.linspace(0, 1, idxs_src[1] - idxs_src[0]),
        y=y_src[idxs_src[0] : idxs_src[1]],
        assume_sorted=True,
    )

    idxs_dst = librosa.time_to_samples((db_dst[0], db_src[num_bars - 1]), sr=sr)

    y_src_warped = mapping(np.linspace(0, 1, idxs_dst[1] - idxs_dst[0]))

    downbeats = np.linspace(
        0, librosa.samples_to_time(len(y_src_warped), sr=sr), num_bars
    )

    return (
        y_src_warped,
        y_dst[idxs_dst[0] : idxs_dst[1]],
        downbeats,
        idxs_src,
        idxs_dst,
    )
