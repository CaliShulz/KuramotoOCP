
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.numeric import outer
from numpy.lib.index_tricks import nd_grid
from dedalus import public as de
from useful_functions_flow import *
from OCP_Kuramoto_flow import *

# importing os module
import os

# Parent Directory path
parent_dir = os.getcwd()

import pickle

directory = "09_09_02_48_54_SIM"
# create a binary pickle file 
file_data = directory + "/data.pkl"

# test open data
loaded_data = pickle.load( open( file_data, "rb" ) )
save_output_fig(directory,loaded_data)
