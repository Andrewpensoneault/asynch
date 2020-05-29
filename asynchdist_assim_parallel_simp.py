import sys
from mpi4py import MPI
import numpy as np
from asynch_py.io import get_asynch_params, get_forcings, create_output_folders, write_sav, read_ini, create_ini, make_forcing_files, create_gbl, remove_files, write_results, get_ids, create_tmp_folder 
import datetime
import time as ttt
from asynch_py.asynch_interface_parallel import asynchsolver

#Initialize MPI
cfull = MPI.COMM_WORLD
nproc = cfull.Get_size()
my_rank = cfull.Get_rank()

color = my_rank
key = 0

csmall = cfull.Split(color,key)
small_rank = csmall.Get_rank()

buffer_size = 100 #size of one sprintf entry for output
#(nproc, my_rank, csmall, cfull) = Initialize_MPI()
#asynch_data = Initialize_Ensemble(sys.argv[1])
#large_string_len = Get_String_Size(buffer_size,asynch_data)
#cfull.Barrier()
#tmax, tcurrent = get_time_range(asynch_data)
#write_gbls(asynch_data)
#while tcurrent<tmax:
#    if my_rank == 0:
#        write_inis(asynch_data)
#        write_str(asynch_data)



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

asynch_dict = {}

my_ens_num = int(np.ceil(float(ens_num)/float(nproc)))
ens_list = np.array([i for i in range(my_rank*my_ens_num,(my_rank+1)*my_ens_num)])

large_string_len = buffer_size*num_steps*link_num*link_var_num

if my_rank == 0:
    create_output_folders(asynch_data)
    write_sav(asynch_data)
    forcing_data = get_forcings(asynch_data)
    for dsource in asynch_data['measure']:
        dsource.get_meas(asynch_data['time_window'])
cfull.Barrier()

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

cfull.Barrier()
for n in ens_list:
    if n < ens_num:
        n_idx = np.nonzero(ens_list==n)[0][0]
        create_gbl(n, global_params[:,n_idx], asynch_data, current_time, time_step, num_steps)
        create_ini(asynch_data['tmp_folder']+str(n)+'.ini',init_id_list,link_var_num,asynch_data['model_num'],init_cond[:,n_idx])
        asynch_dict[n] = asynchsolver(csmall,small_rank)
        create_tmp_folder(asynch_data,n)
        asynch_dict[n].Parse_GBL(str(asynch_data['tmp_folder'] + str(n) + '.gbl'))
        asynch_dict[n].Load_Network()
        asynch_dict[n].Partition_Network()
        asynch_dict[n].Load_Network_Parameters(False)
        asynch_dict[n].Load_Dams()
        asynch_dict[n].Load_Numerical_Error_Data()
        asynch_dict[n].Initialize_Model()
        asynch_dict[n].Load_Save_Lists() 

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
    cfull.Barrier()
    for n in ens_list:
        if n < ens_num:
            n_idx = np.nonzero(ens_list==n)[0][0]
            if i==0:
                asynch_dict[n].Load_Initial_Conditions()
            else:
                param_next = state_last[-num_param:,n_idx]
                init_next = state_last[-(num_param+num_links*link_var_num):-num_param,n_idx]
                init = [[init_next[k+j*link_var_num] for k in range(link_var_num)] for j in range(num_links)]
                
                asynch_dict[n].Set_System_State(0,init)
                asynch_dict[n].Set_Global_Parameters(param_next.tolist())
            asynch_dict[n].Load_Forcings()
            asynch_dict[n].Finalize_Network()
            asynch_dict[n].Calculate_Step_Sizes()
            
            #Prepare outputs
            asynch_dict[n].Prepare_Output()
            
            #Advance solver
            try:
                asynch_dict[n].Prepare_Temp_Files()
                asynch_dict[n].Advance(True)
                output_string = asynch_dict[n].Create_Local_Output(None,large_string_len)
                parameters = np.expand_dims(np.array(asynch_dict[n].Get_Global_Parameters()[0]),1)
                outarray = np.fromstring(output_string,sep=',')[:-1] #removes extra comma introduced in writing csv
                outvec = np.expand_dims(np.reshape(outarray,(num_steps+1,-1))[1:,:].flatten(),1)
                state[:,n_idx] = np.concatenate((outvec,parameters)).flatten()
            except:
                print('failed on ' + str(n) +  ', retrying')
                asynch_dict[n].Prepare_Temp_Files()
                asynch_dict[n].Advance(True)
                output_string = asynch_dict[n].Create_Local_Output(None,large_string_len)
                parameters = np.expand_dims(np.array(asynch_dict[n].Get_Global_Parameters()[0]),1)
                outarray = np.fromstring(output_string,sep=',')[:-1] #removes extra comma introduced in writing csv
                outvec = np.expand_dims(np.reshape(outarray,(num_steps+1,-1))[1:,:].flatten(),1)
                state[:,n_idx] = np.concatenate((outvec,parameters)).flatten() 
    
    sendvbuf = state.T.flatten()
    recvbuf = cfull.gather(sendvbuf, root=0)
    i += 1
    if my_rank == 0:
        state_full_flatten = np.array(recvbuf).flatten()
        state_full = np.zeros((state.shape[0],my_ens_num*nproc))
        for num in range(my_ens_num*nproc):
            state_full[:,num] = state_full_flatten[num*state.shape[0]:(num+1)*state.shape[0]]
        state_full = state_full[:,0:ens_num]
        for pert in asynch_data['var_perturb_type']:
            state_full[:-num_param,:] = pert.perturb(state_full[:-num_param,:])
        for pert in asynch_data['param_perturb_type']:
            state_full[-num_param:,:] = pert.perturb(state_full[-num_param:,:])

        if i == 1: 
            current_time += time_step*60*num_steps 
            time = [current_time - ((num_steps-1)-tt)*60*time_step for tt in range(num_steps)]
            data = state_full
        else:
            current_time += time_step*60 
            time = [current_time] 
            data = state_full[-(num_param+num_links*link_var_num):-num_param,:]
        
        for dsource in asynch_data['measure']:
            dsource.get_current_meas(current_time - (num_steps-1)*time_step*60, id_list, link_var_num, num_steps, time_step)
       
        ens_anal = asynch_data['assim'].assimilate(state_full,asynch_data,asynch_data['measure'])
        weight = asynch_data['assim'].weights
        write_results(data,id_list,link_var_num, time,asynch_data,weight) 
        t1 = ttt.time()
    sendvbuf = None
    recvbuf = np.empty((my_ens_num*(num_param+link_var_num*link_num*num_steps)))
    if my_rank == 0:
        state_full = np.concatenate((state_full,np.zeros((state_full.shape[0],my_ens_num*nproc-ens_num))),axis=1)
        sendvbuf = state_full.T.flatten() 
    cfull.Scatter(sendvbuf, recvbuf, root=0)
    state = np.reshape(recvbuf,(-1,my_ens_num),order='f')
    state_last = state[-(num_param+link_num*link_var_num):,:]
if my_rank == 0:
    remove_files(asynch_data['tmp_folder'])
