import numpy as np
import csv
import pandas as pd

acts  = pd.read_csv('holdout_activations_dim1000_lastlayer.csv', index_col=None, header = None)
soft  = pd.read_csv('holdout_softmax.csv', index_col=None, header = None)

frames = [acts, soft]
result = pd.concat(frames, axis = 1)

result.to_csv('concat_act_soft.csv', index_col=None, header = None, index = False)