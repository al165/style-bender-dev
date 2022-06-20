import numpy as np
from tqdm import tqdm
import ot

import matplotlib.pyplot as plt


class SmoothTransform:
    def __init__(self, bary_func=ot.bregman.barycenter_stabilized, 
                 loss_mat=None, steps=10, reg=0.05):
        self.bary_func = bary_func
        self.reg = reg
        # n is not defined (is the length of the activation vector)
        self.loss_mat = loss_mat
    
        self.steps = steps
        self.weights = np.linspace(0.0, 1.0, self.steps+2)[1:-1] # remove w=0 and w=1

        
    def transform(self, source, target, **bary_func_args):
        n = len(source)
        if self.loss_mat is None:
            self.loss_mat = ot.utils.dist0(n)
        A = np.vstack((source, target)).T
        out = np.ones((self.steps, n))    
        for i, w in tqdm(enumerate(self.weights), total=self.steps):
            out[i] = self.bary_func(A, self.loss_mat, reg=self.reg, weights=[w, 1-w], 
                                    numItermax=50000, **bary_func_args)
        return out
    
    @staticmethod
    def plot_transform(trans_mat, source=None, target=None, **plot_args):
        if source is not None and target is not None:
            trans_mat = np.vstack([target, trans_mat, source])
        
        plt.figure(figsize=(10, 10))
        for i, v in enumerate(trans_mat):
            plt.plot(trans_mat[i]-(i+1)/10, **plot_args)
        plt.show()