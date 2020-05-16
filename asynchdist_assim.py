from asynch_py.asynch_interface import asynchsolver
import sys
from mpi4py import MPI
import numpy as np

## Parse command line arguments
numargs = len(sys.argv)
if numargs != 2:
	print 'Need an input .gbl file'
	sys.exit(1)

## Make a json file that gets initial numbers
#get_asynch_params()

#Should contain the following context
#1). Number of steps in a model run
#2). How long one step will be
#3). Start and stop time window
#4). filename of a txt file of the form of a gbl file missing the following:
#sav
#str
#rvr
#prm
#output
#5-6). filename for rvr and prm 
#7). Source for rainfall 
#8). Source for evaporation
#9). Method for Assimilation
#10). Respective Assimilation Parameters
#11). List of Method for Perturbation
#12). Respective Perturbation Parameters
#13). Method for Measurement
#14). Respective Measurement Parameters
#15). Output Folder













buffer_size = 20
num_steps = 48
num_links = 3202
link_vars = 4
t_steps = 50
ens_num = 1
param_init = np.array([0.33, 0.20, -0.1, 0.02, 2.0425e-6, 0.02, 0.5, 0.10, 0.0, 99.0, 3.0])     
large_string_len = buffer_size*num_steps*num_links*num_vars
num_param = len(param_init)

ens_bg = np.zeros((link_vars*num_steps*num_links+num_param,ens_num))
#get_forcings()

for i in range(tsteps):
#    get_current_forcings()
    for n in ens_to_assim:
#        make_files(n) 
        if i==0:
            asynch = asynchsolver()
            asynch.Parse_GBL(sys.argv[1] + str(n) + '.gbl')
            asynch.Load_Network()
            asynch.Partition_Network()
            asynch.Load_Network_Parameters(False)
            asynch.Load_Dams()
            asynch.Load_Numerical_Error_Data()
            asynch.Initialize_Model()
            asynch.Load_Save_Lists() 
            asynch.Load_Initial_Conditions()
        else:
#            init_next = ens_anal[-param_num:,n]
#            param_next = ens_anal[-(param_num+num_links*link_vars):-param_num,n]
            asynch.Set_System_State(0,init_next)
            asynch.Set_Global_Parameters(param_next)
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
        parameters = asynch.Get_Global_Parameters() 
        outarray = np.fromstring(output_string,sep=',')[:-1] #removes extra comma introduced in writing csv
        outvec = np.reshape(outarray,(numsteps+1,-1))[1:,:].flatten()
        state = np.concatenate((outvec,parameters))
#        state += perturb(state)
        ens_bg[:,n] = state
#    ens_anal = assimilate()
#    write_results() 
#remove_files()
