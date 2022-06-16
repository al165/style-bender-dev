import os
import sys
import pickle as pkl

import numpy as np

import torch

import librosa
from madmom.features import DBNDownBeatTrackingProcessor, RNNDownBeatProcessor

sys.path.extend(["../AUDIO_SEPARATION", "../COMMON_UTILS/"])

from drum_decomposition import (
    drumGridFromHs,
    quantiseGrids,
    splitActivationsAndAudio,
    isolateSources,
    getDecomposition,
    getSamples,
)


DEFAULT_DECOMPOSITION_KWARGS = {
    "R": 3,
    "T": 10,
    "H": None,
    "W": None,
    "beta": 1.5,
    "max_iter": 1000,
    "tol": 1e-5,
    "sW": None,
    "sH": None,
    "sparse_fit": False,
    "device": "cpu",
}


DEFAULT_NAMES = [
    "KD",
    "SD",
    "HH",
]


def getDownbeats(fp: str, transition_lambda: float = 16, **kwargs) -> np.array:
    proc = DBNDownBeatTrackingProcessor(
        4, fps=100, transition_lambda=transition_lambda, **kwargs
    )
    rnnproc = RNNDownBeatProcessor()
    act = rnnproc(fp)

    beats = proc(act)
    beats = beats[np.where(beats[:, 1] == 1)[0][0] :]  # start from first downbeat
    beats = beats[: np.where(beats[:, 1] == 1)[0][-1] + 1]  # until last first downbeat

    downbeats = beats[np.where(beats[:, 1] == 1)][:, 0]

    return downbeats


class DrumProcessor:
    """
    DrumProcessor

    Processes an isolated drum audio track and obtains the following
    file structure and files:
    ```
        - drum_grids.pkl
        - drum_samples/
            BD/
                BD_xx.wav
            KD/...
            HH/...
        - drum_tracks/
            bd.wav
            kd.wav
            hh.wav
        - drums.wav
    ```

    """

    @staticmethod
    def process(
        fp: str = None,
        y: np.array = None,
        sr: int = 22050,
        downbeats: np.array = None,
        verbose: bool = False,
        **kwargs,
    ) -> dict:
        """
        Process an audio track

        Parameters
        ----------
        fp : str (optional)
            File path to drum audio
        y : np.array (optional)
            Audio data.
        sr : int (optional)
            Sample rate if `y` is provided. Default 22050.
        downbeats : np.array (optional)
            If `fp` is not provided, then must pass an array of downbeat start times.
        verbose : bool (optional)
            Print progress. Default False.
        **kwargs
            Arguments to send to `getDecomposition`.

        Returns
        -------
        data : dict
            Computed data containing:
                - "ys": list[np.array] of isolated sources
                - "net": NMFD model
                - "samples": list[np.array] of audio samples
                - "grids": np.array of beat grids
                - "bars": list[np.array] of sliced and separated audio
        """

        if fp is None and downbeats is None:
            raise ValueError("must provide `downbeats` if `fp` is not provided")

        if downbeats is None:
            if verbose:
                print("computing downbeats")

            downbeats = getDownbeats(fp)

        if y is None:
            if verbose:
                print(f"loading {fp}")
            y, sr = librosa.load(fp, sr=sr)

        if "device" not in kwargs:
            device = "cpu"
        else:
            device = kwargs["device"]

        kwargs = {**DEFAULT_DECOMPOSITION_KWARGS, **kwargs}

        if verbose:
            print("decomposing audio")
        W, H, V, phi, net = getDecomposition(y, **kwargs)

        if verbose:
            print("isolating sources")
        ys = isolateSources(net, phi, device)

        H_bars, bars = splitActivationsAndAudio(ys, H, downbeats, sr=sr, delay=1)

        if verbose:
            print("computing drum grids")
        grids = drumGridFromHs(H_bars, grid_size=48)
        grids = quantiseGrids(grids)

        if verbose:
            print("creating samples")
        samples = [getSamples(net, i) for i in range(kwargs["R"])]

        data = {
            "ys": ys,
            "sr": sr,
            "net": net,
            "samples": samples,
            "grids": grids,
            "bars": bars,
        }

        return data

    __call__ = process

    @staticmethod
    def getInitialTemplates(fp: str, device: str = "cpu"):
        with open("../AUDIO_SEPARATION/drum_templates.pkl", "rb") as f:
            templates = pkl.load(f)
            kd_temp = templates["kd_temp"]
            sd_temp = templates["sd_temp"]
            hh_temp = templates["hh_temp"]

        W_0 = torch.from_numpy(np.stack([kd_temp, sd_temp, hh_temp], axis=1))

        return W_0

    @staticmethod
    def saveOutput(data: dict, base: str, names: list = DEFAULT_NAMES):
        """
        - drum_grids.pkl
        - drum_samples/
            00_BD/
                00_BD_xx.wav
            01_KD/...
            02_HH/...
        - drum_tracks/
            00_KD.wav
            01_SD.wav
            03_HH.wav
        - drums.wav

        Parameters
        ----------
        data : dict
        base : str
            Root dir to save data
        """

        import scipy.io.wavfile as wave

        os.makedirs(os.path.join(base, "drum_samples"), exist_ok=True)
        os.makedirs(os.path.join(base, "drum_tracks"), exist_ok=True)

        # save grids
        with open(os.path.join(base, "drum_grids.pkl"), "wb") as f:
            pkl.dump(data["grids"], f)

        # save isolated tracks
        for i, y in enumerate(data["ys"]):
            name = f"{i:02d}"
            if i < len(names):
                name += "_" + names[i]

            wave.write(os.path.join(base, "drum_tracks", name + ".wav"), data["sr"], y)

        # save samples
        for i, ins in enumerate(data["samples"]):
            name = f"{i:02d}"
            if i < len(names):
                name += "_" + names[i]

            os.makedirs(os.path.join(base, "drum_samples", name.upper()), exist_ok=True)
            for j, y in enumerate(ins):
                wave.write(
                    os.path.join(
                        base, "drum_samples", name.upper(), f"{name}_{j:03d}" + ".wav"
                    ),
                    data["sr"],
                    y,
                )
