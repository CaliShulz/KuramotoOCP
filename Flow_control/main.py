# Linear Quadratic OCP for Heat Equation with drift

# seems to work reasonably well
# gradient norm do not decrease enough..

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

N_grids    = [128]
dts        = [0.005, 0.001]
T_finals   = [10]
betas      = [5*10**(-3) , 1*10**(-5)]
diff_s     = [0.25]
phase_lags = [0]
max_iters  = [50,100]
K_int      = 1

from datetime import datetime
import pickle


for N_grid in N_grids:
    for dt in dts:
        for T_final in T_finals:
            for beta in betas:
                for diff in diff_s:
                    for phase_lag in phase_lags:
                        for max_iter in max_iters:

                            start = datetime.now() # current date and time
                            start_time = start.strftime("%m_%d_%H_%M_%S")

                            directory  = start_time +"_SIM"

                            path       = os.path.join(parent_dir, directory)
                            os.mkdir(path)


                            param_dict = {}
                            param_dict['N_grid']    = N_grid
                            param_dict['max_it']    = max_iter
                            param_dict['dt']        = dt
                            param_dict['T_final']   = T_final
                            param_dict['beta']      = beta
                            param_dict['diffusion'] = diff
                            param_dict['phase_lag'] = phase_lag
                            param_dict['K_int']     = K_int

                            
                            output_data = launch_sim(param_dict)

                            finish = datetime.now() # current date and time
                            finish_time = finish.strftime("%m/%d/%Y, %H:%M:%S")

                            file_name = directory + "/report.txt"
                            with open(file_name, 'w') as f: 

                                f.write(directory + '\n')
                                f.write('\n')
                                f.write('Start Time' + start_time + '\n')
                                f.write('Finish Time' + finish_time + '\n')

                                for key, value in param_dict.items(): 
                                    f.write('%s : %s\n' % (key, value))

                            # create a binary pickle file 
                            file_data = directory + "/data.pkl"
                            with open(file_data,"wb") as f:
                                # write the python object (dict) to pickle file
                                pickle.dump(output_data,f)

                            # test open data
                            with open(file_data, "rb") as input_file:
                                loaded_data = pickle.load(input_file)
                                save_output_fig(directory,loaded_data)
