from asynch_py.asynch_interface import asynchsolver
import sys
from mpi4py import MPI
import numpy as np
buffer_size = 20 #size of one sprintf entry for output

## Parse command line arguments
numargs = len(sys.argv)
if numargs != 2:
	print 'Need an input .json file'
	sys.exit(1)

## Make a json file that gets initial numbers
#Should contain the following context
#1). Number of steps in a model run
#2). How long one step will be
#3). Initial Global Parameters
#4). Initial initial Condition
#5). Start and stop time window
#6). filename of a txt file of the form of a gbl file missing the following:
#sav
#str
#rvr
#prm
#output
#7-8). filename for rvr and prm 
#9). Source for rainfall 
#10). Source for evaporation
#11). Method for Assimilation
#12). Respective Assimilation Parameters
#13). List of Method for Perturbation
#14). Respective Perturbation Parameters
#15). Method for Measurement
#16). Respective Measurement Parameters
#17). Output Folder

asynch_data = get_asynch_params(sys.argv[1])
import pdb; pdb.set_trace()
#forcing_data = get_forcings()
#measure_data = get_meas()

large_string_len = buffer_size*num_steps*num_links*num_vars
ens_bg = np.zeros((link_vars*num_steps*num_links+num_param,ens_num))
for i in range(tsteps):
#    forcings = get_current_forcings(forcing_data)
    for n in ens_to_assim:
#        make_files(n,asynch_data,forcings) 
        if i==0:
            asynch = asynchsolver()
            asynch.Parse_GBL(data['tmp_folder'] + str(n) + '.gbl')
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
#        state += perturb(state,asynch_data)
        ens_bg[:,n] = state
#    measure = get_meas(asynch_data)
#    ens_anal = assimilate(state,asynch_data)
#    write_results(asynch_data) 
#remove_files()
