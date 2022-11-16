import numpy as np
#from scipy.signal import find_peaks
from findpeaks import findpeaks

import ot

from tqdm.auto import tqdm


class SmoothTransition:
    def __init__(self, src, trg, src_scale=None, trg_scale=None):
        assert src.shape == trg.shape

        if src_scale is None:
            src_scale = src.max(axis=1).reshape((-1, 1))
        self.src_scale = src_scale

        if trg_scale is None:
            trg_scale = trg.max(axis=1).reshape((-1, 1))
        self.trg_scale = trg_scale

        self.src = src.copy() / self.src_scale
        self.trg = trg.copy() / self.trg_scale

        self.src_peaks = dict()
        self.trg_peaks = dict()

        for i in range(len(src)):
            pi, ph, ps = self.get_peak_data(np.insert(self.src[i], 0, 0))
            for k in pi.keys():
                pi[k] -= 1
            self.src_peaks[i] = (pi, ph, ps)

            # save a typical peak for use in creating new peaks
            #candidate_peaks_idxs = np.where(ph > ph.max()/2)
            #
            #self.src_typical_peak

            pi, ph, ps = self.get_peak_data(np.insert(self.trg[i], 0, 0))
            for k in pi.keys():
                pi[k] -= 1
            self.trg_peaks[i] = (pi, ph, ps)



    @staticmethod
    def get_peak_data(x):
        fp = findpeaks(lookahead=3, method='topology', verbose=0)
        df = fp.fit(x)['df']
        peaks = df.groupby('labx')

        peak_heights = peaks['y'].max()
        peak_score = peaks['score'].max()
        peak_idxs = peaks.indices

        return peak_idxs, peak_heights, peak_score


    def transform(self, t, power=1, score_threshold=0.2):
        t = min(max(t, 0.0), 1.0)

        #if t == 0.0:
        #    return self.src * self.src_scale
        #elif t == 1.0:
        #    return self.trg * self.trg_scale

        t_s = np.power(t, power)
        t_t = np.power(1-t, power)

        acts = []
        for i in range(len(self.src)):
            act_t = np.zeros_like(self.src[i])

            pi, ph, ps = self.src_peaks[i]
            for j in pi.keys():
                if ps[j] < score_threshold:
                    continue

                if ph[j] <= t_s:
                    continue

                act_t[pi[j]] += self.src[i][pi[j]]


            pi, ph, ps = self.trg_peaks[i]
            for j in pi.keys():
                if ps[j] < score_threshold:
                    continue

                if ph[j] <= t_t:
                    continue

                act_t[pi[j]] += self.trg[i][pi[j]]          

            acts.append(act_t)

        return np.stack(acts) * self.trg_scale
    
    __call__ = transform
            
    @staticmethod
    def add_peak(a, p, h):
        peak_h, idx_offset = SmoothTransition.makePeak(h)

        start = p - idx_offset
        if start < 0:
            peak_h = peak_h[-start:]
            start = 0

        length = min(len(peak_h), len(a) - start)    
        end = start + length

        a[start:end] += peak_h[:length]
        return a
    
    @staticmethod
    def makePeak(height):
        atk = np.power(np.linspace(0, 1, 3), 6) * height
        dec = np.power(np.linspace(1, 0, 4), 6) * height

        peak = np.concatenate([atk, dec[1:]])
        return peak, len(atk) - 1



class SmoothTransform:
    def __init__(
        self,
        bary_func=ot.bregman.barycenter_stabilized,
        loss_mat=None,
        steps=10,
        reg=0.05,
    ):
        self.bary_func = bary_func
        self.reg = reg
        # n is not defined (is the length of the activation vector)
        self.loss_mat = loss_mat

        self.steps = steps
        self.weights = np.linspace(0.0, 1.0, self.steps) #[1:-1]  # remove w=0 and w=1

    def transform(self, source, target, **bary_func_args):
        n = len(source)
        if self.loss_mat is None:
            self.loss_mat = ot.utils.dist0(n)
        A = np.vstack((source, target)).T
        out = np.ones((self.steps, n))
        for i, w in tqdm(enumerate(self.weights), total=self.steps):
            if i == 0:
                out[i] = source
            elif i == self.steps - 1:
                out[i] = target
            else:
                out[i] = self.bary_func(
                    A,
                    self.loss_mat,
                    reg=self.reg,
                    weights=[1 - w, w],
                    numItermax=50000,
                    **bary_func_args
                )
        return out

    @staticmethod
    def plot_transform(trans_mat, source=None, target=None, ax=None, **plot_args):
        import matplotlib.pyplot as plt
        
        if source is not None and target is not None:
            trans_mat = np.vstack([target, trans_mat, source])

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))

        for i, v in enumerate(trans_mat):
            ax.plot(trans_mat[i]/trans_mat[i].max() - (i*1.1), **plot_args)
            
        ax.set_yticks([])
        
        return ax


### EXAMPLE
# src, dest = joblib.load("activations.pkl")

# k = 300
# ps, qs = src[0][:k], dest[0][:k]
# ps, qs = ps/ps.sum(), qs/qs.sum()

# sm = SmoothTransform()
# trans = sm.transform(ps, qs)

# sm.plot_transform(trans, source=ps, target=qs)
