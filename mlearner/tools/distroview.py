import sys
import numpy as np
import matplotlib.pyplot as plt

def _err_(msg):
    print >> sys.stderr, msg

class NoiseVector:
    def __init__(self, mu, sigma_l, dim):
        self.mu = mu
        self.sigma_l = sigma_l
        self.dim = dim

    def set_xy(self,x,y):
        self.x = x
        self.y = y

    def add_noise(self, n):
        dist_d = {}
        for sigma in self.sigma_l:
            sig_dist_l = []
            for _ in range(n):
                X = self.x + np.random.normal(self.mu, sigma, self.dim)
                Y = self.y + np.random.normal(self.mu, sigma, self.dim)
                sig_dist_l.append(np.linalg.norm(X - Y))
            dist_d[sigma] = sig_dist_l
        return dist_d

    def gset_mean_rdist(self, n):
        self.dist_d = self.add_noise(n)
        self.avg_dist_d = {}
        for sigma in self.sigma_l:
            self.avg_dist_d[sigma] = np.average(self.dist_d[sigma])
        return self.avg_dist_d

    def plot_view(self):
        plt.figure(1)
        plt.hold(True)
        for i in range(len(self.dist_d.values()[0])):
            plt.scatter(self.sigma_l, [self.dist_d[sigma][i] for sigma in self.sigma_l], c='b', marker='o')
        plt.plot(self.sigma_l, [self.avg_dist_d[sigma] for sigma in self.sigma_l], 'r--', linewidth=1)
        plt.show()
