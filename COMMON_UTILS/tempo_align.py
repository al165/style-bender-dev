import warnings

import numpy as np
from scipy.interpolate import interp1d

import librosa


def matchAudioEvents(
    y: np.ndarray,
    sr: int,
    src_events: np.ndarray,
    dst_events: np.ndarray,
    units: str = "time",
    hq: bool = False,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Warps audio such that `src_events` align with `dst_events`.

    Parameters
    ----------
    y: np.ndarray
    sr: int
    src_events: np.ndarray
    dst_events: np.ndarray
    units: str = "time"
        Temporal units of `src_` and `dst_events`. Must be one of
        {"time", "frames", "samples"}.
    hq: bool = False
    hop_length: int = 512
        Only required if `units="frames"`.

    Returns
    -------
    y_warped : np.ndarray
    """

    if len(src_events) != len(dst_events):
        raise ValueError(
            f"`src_events` (len={len(src_events)}) not equal to `dst_events` (len={len(dst_events)})."
        )

    if hq:
        try:
            import pyrubberband
        except ImportError:
            warnings.warn(
                "`hq=True` requires `pyrubberband` (not found), setting `hq=False`"
            )
            hq = False

    if units == "time":
        src_bounds = librosa.time_to_samples(src_events, sr=sr)
        dst_bounds = librosa.time_to_samples(dst_events, sr=sr)
    elif units == "samples":
        src_bounds = int(src_events)
        dst_bounds = int(dst_events)
    elif units == "frames":
        src_bounds = librosa.frames_to_samples(src_events, hop_length=hop_length)
        dst_bounds = librosa.frames_to_samples(dst_events, hop_length=hop_length)
    else:
        raise ValueError("`units` must be one of {'time', 'samples', 'frames'}")

    src_segments = np.split(y, src_bounds)
    dst_widths = np.diff(dst_bounds, prepend=0, append=len(y))

    y_warped = []
    for s, d_w in zip(src_segments, dst_widths):
        if d_w == 0:
            continue
        rate = len(s) / d_w

        if rate <= 0:
            s_warped = np.zeros(d_w)
        elif hq:
            s_warped = pyrubberband.pyrb.time_stretch(y=s, sr=sr, rate=rate)
        else:
            s_warped = librosa.effects.time_stretch(y=s, rate=rate)

        y_warped.append(s_warped)

    return np.concatenate(y_warped)


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
        if fix_rate <= 0:
            b_q = np.zeros(len(b))
        elif hq:
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple]:

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

    idxs_dst = librosa.time_to_samples((db_dst[0], db_dst[num_bars - 1]), sr=sr)

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
