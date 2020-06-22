import sys
import numpy as np
import datetime
import time as ttt
from asynch_py.asynch_interface_parallel import Assim

## Parse command line arguments
numargs = len(sys.argv)
if numargs != 2:
    print('Need an input .json file')
    sys.exit(1)

#Initialize Assim
assim = Assim(sys.argv[1])
param_num = len(assim.asynch_data['init_global_params'])
buffer_size = 20*assim.asynch_data['num_steps']*assim.asynch_data['link_var_num']*len(assim.asynch_data['id_list'])
#Create Initial Files
assim.create_output_folders()
assim.full_comm.Barrier()
assim.create_tmp_folders()
assim.full_comm.Barrier()
assim.write_sav()
assim.write_ini()
assim.write_gbl()

#Get Forcings and Data
assim.get_meas()
assim.get_forcings()

#Run Model
t0 = ttt.time()
while assim.current_time < assim.asynch_data['time_window'][1]:
    assim.write_ini()
    if assim.my_rank == 0:
        t1 = ttt.time()
        print('Date: ' + datetime.datetime.utcfromtimestamp(assim.current_time).strftime('%Y-%m-%d %H:%M:%S') + ', Time to run: ' + str(t1-t0) + ' seconds')
        t0 = ttt.time()
    assim.write_forcings()
    assim.full_comm.Barrier()
    state = assim.advance(buffer_size)
    state_full = assim.gather_outputs(state)
    if assim.my_rank == 0:
        assim.get_current_meas()
        weights = assim.asynch_data['assim'].weights
        #(state_full,weights) = assim.assimilate(state_full, assim.asynch_data['measure'])
        state_full = assim.fix_states(state_full)
        assim.write_outputs(state_full,weights)
        print('q value: ' + str(np.min(state_full[0:-param_num:assim.asynch_data['link_var_num'],:]))+' to '+str(np.max(state_full[0:-param_num:assim.asynch_data['link_var_num'],:])))
        print('sp value: ' + str(np.min(state_full[1:-param_num:assim.asynch_data['link_var_num'],:]))+' to '+ str(np.max(state_full[1:-param_num:assim.asynch_data['link_var_num'],:])))
        print('st value: ' + str(np.min(state_full[2:-param_num:assim.asynch_data['link_var_num'],:]))+' to '+ str(np.max(state_full[2:-param_num:assim.asynch_data['link_var_num'],:])))
        print('ss value: ' + str(np.min(state_full[3:-param_num:assim.asynch_data['link_var_num'],:]))+' to '+ str(np.max(state_full[3:-param_num:assim.asynch_data['link_var_num'],:])))
        print('max params: ' + str(np.amax(state_full[-param_num:,:],axis=1)))
        print('min params: ' + str(np.amin(state_full[-param_num:,:],axis=1)))
    state = assim.scatter_outputs(state_full)
    assim.init_cond = state[(-param_num-(assim.num_steps)*assim.link_vars):(-param_num-(assim.num_steps-1)*assim.link_vars),:]
    assim.global_param = state[-param_num:,:]
    assim.first_time = False
assim.remove_files()
