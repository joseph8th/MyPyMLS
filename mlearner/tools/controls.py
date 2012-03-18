from settings import *
from tools.distroview import *
import numpy as np

# Tool controller for studying mach. learnin'

class ToolControl:
    def __init__(self, sett):
        self.sett = sett

    def run_tool(self):
        if self.sett['tool'] == '1':
            x = np.zeros(self.sett['dim'])
            y = np.zeros(self.sett['dim'])
            x[0] = 1.0
            v = NoiseVector(self.sett['mu'], SIGMA_LIST, self.sett['dim'])
            v.set_xy(x,y)
            v.gset_mean_rdist(self.sett['n'])
            v.plot_view()
            
