from asynch_py.asynch_interface_parallel import asynchsolver
import sys
from mpi4py import MPI
import numpy as np
from asynch_py.io import get_asynch_params, get_forcings, create_output_folders, write_sav, read_ini, create_ini, make_forcing_files, create_gbl, remove_files, write_results, get_ids 
import datetime
import time as ttt

comm = MPI.COMM_WORLD
nproc = comm.Get_size()
my_rank = comm.Get_rank()
import pdb; pdb.set_trace()
buffer_size = 100 #size of one sprintf entry for output
## Parse command line arguments
numargs = len(sys.argv)
if numargs != 2:
	print 'Need an input .json file'
	sys.exit(1)

asynch_data = get_asynch_params(sys.argv[1])
asynch_data['id_list'] = get_ids(asynch_data)
link_num = asynch_data['link_num']
num_steps = asynch_data['num_steps']
link_var_num = asynch_data['link_var_num']
ens_num = asynch_data['assim_params']['ens_num']
num_param = len(asynch_data['init_global_params'])
time_start = asynch_data['time_window'][0]
time_stop = asynch_data['time_window'][1]
current_time = time_start
time_step = asynch_data['step_size']
num_links = len(asynch_data['id_list'])
id_list = asynch_data['id_list']

my_ens_num = int(np.ceil(float(ens_num)/float(nproc)))
ens_list = np.array([i for i in range(my_rank*my_ens_num,(my_rank+1)*my_ens_num)])

large_string_len = buffer_size*num_steps*link_num*link_var_num

if my_rank == 0:
    write_sav(asynch_data)
    create_output_folders(asynch_data)
    forcing_data = get_forcings(asynch_data)
    for dsource in asynch_data['measure']:
        dsource.get_meas(asynch_data['time_window'])

(ic, init_id_list) = read_ini(asynch_data['init_cond'])

global_params = np.zeros((num_param,my_ens_num))
init_cond = np.zeros((len(ic),ens_num))
for n in range(my_ens_num):
    init_cond[:,n] = ic 
    global_params[:,n] = asynch_data['init_global_params']
 
for pert in asynch_data['var_perturb_type']:
    init_cond = pert.init_perturb(init_cond)
for pert in asynch_data['param_perturb_type']:
    global_params = pert.init_perturb(global_params)

for n in ens_list:
    if n < ens_num:
        n_idx = np.nonzero(ens_list==n)[0][0]
        create_ini(asynch_data['tmp_folder']+str(n)+'.ini',init_id_list,link_var_num,asynch_data['model_num'],init_cond[:,n_idx])

i = 0
state = np.zeros((len(ic)*num_steps+num_param,my_ens_num))
if my_rank == 0:
    t1 = ttt.time()
    t0 = ttt.time()
while current_time<time_stop:
    if my_rank == 0:
        print('Date: ' + datetime.datetime.utcfromtimestamp(current_time).strftime('%Y-%m-%d %H:%M:%S') + ', Time to run: ' + str(t1-t0) + ' seconds')
        t0 = ttt.time()
        make_forcing_files(forcing_data, asynch_data, current_time)
    comm.Barrier()
    for n in ens_list:
        if n < ens_num:
            n_idx = np.nonzero(ens_list==n)[0][0]
            create_gbl(n, global_params[:,n_idx], asynch_data, current_time, time_step, num_steps)
            if i==0:
                asynch = asynchsolver(comm,my_rank)
                asynch.Parse_GBL(str(asynch_data['tmp_folder'] + str(n) + '.gbl'))
                asynch.Load_Network()
                asynch.Partition_Network()
                asynch.Load_Network_Parameters(False)
                asynch.Load_Dams()
                asynch.Load_Numerical_Error_Data()
                asynch.Initialize_Model()
                asynch.Load_Save_Lists() 
                asynch.Load_Initial_Conditions()
                write_times = num_steps
            else:
                param_next = state[-num_param:,n_idx]
                init_next = state[-(num_param+num_links*link_var_num):-num_param,n_idx]
                init = [[init_next[i+j*link_var_num] for i in range(link_var_num)] for j in range(num_links)]
                
                asynch.Set_System_State(0,init)
                asynch.Set_Global_Parameters(param_next.tolist())
                write_times = 1
            asynch.Load_Forcings()
            asynch.Finalize_Network()
            asynch.Calculate_Step_Sizes()
            
            #Prepare outputs
            asynch.Prepare_Output()
            asynch.Prepare_Temp_Files()
            
            #Advance solver
            asynch.Advance(True)
            
            #Create output files
            output_string = asynch.Create_Local_Output(None,large_string_len)
            parameters = np.expand_dims(np.array(asynch.Get_Global_Parameters()[0]),1)
            outarray = np.fromstring(output_string,sep=',')[:-1] #removes extra comma introduced in writing csv
            outvec = np.expand_dims(np.reshape(outarray,(num_steps+1,-1))[1:,:].flatten(),1)
            state[:,n_idx] = np.concatenate((outvec,parameters)).flatten()
    sendvbuf = state.flatten()
    recvbuf = None
    if my_rank == 0:
        recvbuff = np.empty((nproc,(link_num*num_step*link_var_num+param_num)*my_ens_num))
    com.Gather(sendvbuf, recvbuf, root=0)
    #transform into something usable
    if my_rank == 0:    
        for pert in asynch_data['var_perturb_type']:
            state[:-num_param,:] = pert.perturb(state[:-num_param,:])
        for pert in asynch_data['param_perturb_type']:
            state[-num_param:,:] = pert.perturb(state[-num_param:,:])

        i += 1
        if i == 1: 
            current_time += time_step*60*num_steps 
            time = [current_time - ((num_steps-1)-tt)*60*time_step for tt in range(num_steps)]
            data = state
        else:
            current_time += time_step*60 
            time = [current_time] 
            data = state[-(num_param+num_links*link_var_num):-num_param,:]
        
        for dsource in asynch_data['measure']:
            dsource.get_current_meas(current_time - (num_steps-1)*time_step*60, id_list, link_var_num, num_steps, time_step)
        
        ens_anal = asynch_data['assim'].assimilate(state,asynch_data,asynch_data['measure'])
        weight = asynch_data['assim'].weights
        write_results(data,id_list,link_var_num, time,asynch_data,weight) 
        t1 = ttt.time()
    sendvbuf = None
    recvbuf = np.empty((my_ens_num*(param_num+link_var_num*link_num)))
    if my_rank == 0:
        sendvbuf = flattened_ninit
    com.Scatter(sendvbuf, recvbuf, root=0)
    #transform into something usable
if my_rank == 0:
    remove_files(asynch_data['tmp_folder'])
