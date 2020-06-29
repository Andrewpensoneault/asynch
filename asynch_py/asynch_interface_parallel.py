from asynch_py import ASYNCH_LIBRARY_LOCATION
from ctypes import *
from mpi4py import MPI
import numpy
import sys
from mpi4py import MPI
import numpy as np
from asynch_py.io import get_asynch_params, get_forcings, create_output_folders, write_sav, read_ini, create_ini, make_forcing_files, create_gbl, remove_files, write_results, get_ids, create_tmp_folder
from asynch_py.perturb import fix_states
import datetime
import psutil
import os

#ASYNCH_LIBRARY_LOCATION = '/home/ssma/NewAsynchVersion/libs/libasynch_py.so'

#Predefined values *******************************************************
ASYNCH_BAD_TYPE = -1
ASYNCH_CHAR = 0
ASYNCH_SHORT = 1
ASYNCH_INT = 2
ASYNCH_FLOAT = 3
ASYNCH_DOUBLE = 4

ASYNCH_MAX_QUERY_SIZE = 1024
ASYNCH_MAX_DB_CONNECTIONS = 16
ASYNCH_DB_LOC_TOPO = 0
ASYNCH_DB_LOC_PARAMS = 1
ASYNCH_DB_LOC_INIT = 2
ASYNCH_DB_LOC_QVS = 3
ASYNCH_DB_LOC_RSV = 4
ASYNCH_DB_LOC_HYDROSAVE = 5
ASYNCH_DB_LOC_PEAKSAVE = 6
ASYNCH_DB_LOC_HYDRO_OUTPUT = 7
ASYNCH_DB_LOC_PEAK_OUTPUT = 8
ASYNCH_DB_LOC_SNAPSHOT_OUTPUT = 9
ASYNCH_DB_LOC_FORCING_START = 10

#Data structures **********************************************************
class FILE(Structure): pass
class RKMethod(Structure): pass
class RKSolutionList(Structure): pass
class ErrorData(Structure): pass
class ConnData(Structure): pass
class Forcing(Structure): pass
class TempStorage(Structure): pass
class io(Structure): pass
class Link(Structure): pass
class UnivVars(Structure): pass
class ForcingData(Structure): pass

class VEC(Structure):
	_fields_ = [("ve",POINTER(c_double)),("dim",c_uint)]

class MAT(Structure):
	_fields_ = [("array",POINTER(c_double)),("me",POINTER(POINTER(c_double))),("m",c_uint),("n",c_uint)]

class QVSData(Structure):
	_fields_ = [("points",POINTER(POINTER(c_double))),("n_values",c_uint)]

ASYNCH_F_DATATYPE = CFUNCTYPE(None,c_double,POINTER(VEC),POINTER(POINTER(VEC)),c_ushort,POINTER(VEC),POINTER(c_double),POINTER(QVSData),POINTER(VEC),c_int,c_void_p,POINTER(VEC))
ASYNCH_RKSOLVER_DATATYPE = CFUNCTYPE(c_int,POINTER(Link),POINTER(UnivVars),POINTER(c_int),c_short,POINTER(FILE),POINTER(ConnData),POINTER(POINTER(Forcing)),POINTER(TempStorage))
ASYNCH_CONSISTENCY_DATATYPE = CFUNCTYPE(None,POINTER(VEC),POINTER(VEC),POINTER(VEC))
ASYNCH_ALG_DATATYPE = CFUNCTYPE(None,POINTER(VEC),POINTER(VEC),POINTER(VEC),POINTER(QVSData),c_int,c_void_p,POINTER(VEC))
ASYNCH_STATECHECK_DATATYPE = CFUNCTYPE(c_int,POINTER(VEC),POINTER(VEC),POINTER(VEC),POINTER(QVSData),c_uint)

ASYNCH_OUTPUT_DATATYPE = CFUNCTYPE(None,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_int,c_void_p)
ASYNCH_OUTPUT_INT_DATATYPE = CFUNCTYPE(c_int,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_int,c_void_p)
ASYNCH_OUTPUT_DOUBLE_DATATYPE = CFUNCTYPE(c_double,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_int,c_void_p)

ASYNCH_PEAKOUTPUT_DATATYPE = CFUNCTYPE(None,c_uint,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_double,c_uint,c_void_p,POINTER(c_char))

class UnivVars(Structure):
	_fields_ = [("type",c_ushort),("method",c_ushort),("max_s",c_ushort),("max_parents",c_ushort),
			("iter_limit",c_int),("max_transfer_steps",c_int),("maxtime",c_double),("t_0",c_double),
			("discont_size",c_uint),("max_localorder",c_uint),("uses_dam",c_ushort),("global_params",POINTER(VEC)),
			("params_size",c_uint),("dam_params_size",c_uint),("disk_params",c_uint),("area_idx",c_uint),
			("areah_idx",c_uint),("rain_filename",c_char_p),("init_filename",c_char_p),("rvr_filename",c_char_p),
			("prm_filename",c_char_p),("init_flag",c_ushort),("rvr_flag",c_ushort),("prm_flag",c_ushort),
			("output_flag",c_ushort),("temp_filename",c_char_p),("dam_filename",c_char_p),("print_time",c_double),
			("print_par_flag",c_ushort),("dam_flag",c_ushort),("hydrosave_flag",c_ushort),("peaksave_flag",c_ushort),
			("hydrosave_filename",c_char_p),("peaksave_filename",c_char_p),("peakfilename",c_char_p),("max_dim",c_uint),
			("outletlink",c_uint),("string_size",c_uint),("query_size",c_uint),("rkd_flag",c_short),
			("convertarea_flag",c_ushort),("discont_tol",c_double),("min_error_tolerances",c_uint),("num_forcings",c_uint),
			("hydros_loc_flag",c_ushort),("peaks_loc_flag",c_ushort),("dump_loc_flag",c_ushort),("res_flag",c_ushort),
			("hydros_loc_filename",c_char_p),("peaks_loc_filename",c_char_p),("dump_loc_filename",c_char_p),("rsv_filename",c_char_p),
			("init_timestamp",c_uint),("res_forcing_idx",c_ushort),("num_states_for_printing",c_uint),("num_print",c_uint),
			("print_indices",POINTER(c_uint)),
			("outputs_d",POINTER(CFUNCTYPE(c_double,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_int,c_void_p))),
			("outputs_i",POINTER(CFUNCTYPE(c_int,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_int,c_void_p))),
			("output_names",POINTER(c_char_p)),("output_specifiers",POINTER(c_char_p)),("output_types",POINTER(c_short)),("output_sizes",POINTER(c_short)),
			("output_data",POINTER(io)),("peakflow_function_name",c_char_p),
			("peakflow_output",CFUNCTYPE(None,c_uint,c_double,POINTER(VEC),POINTER(VEC),POINTER(VEC),c_double,c_uint,c_void_p,c_char_p)),
			("hydro_table",c_char_p),("peak_table",c_char_p),("dump_table",c_char_p)]

class Link(Structure):
	_fields_ = [("method",POINTER(RKMethod)),("list",POINTER(RKSolutionList)),("errorinfo",POINTER(ErrorData)),("params",POINTER(VEC)),
			("f",ASYNCH_F_DATATYPE),
			("alg",ASYNCH_ALG_DATATYPE),
			("state_check",ASYNCH_STATECHECK_DATATYPE),
			("Jacobian",CFUNCTYPE(None,c_double,POINTER(VEC),POINTER(POINTER(VEC)),c_ushort,POINTER(VEC),POINTER(c_double),POINTER(VEC),POINTER(MAT))),
			("RKSolver",ASYNCH_RKSOLVER_DATATYPE),
			("CheckConsistency",ASYNCH_CONSISTENCY_DATATYPE),
			("h",c_double),("last_t",c_double),("print_time",c_double),("next_save",c_double),
			("ID",c_uint),("location",c_uint),("ready",c_short),("numparents",c_ushort),
			("disk_iterations",c_int),("peak_time",c_double),("peak_value",POINTER(VEC)),("parents",POINTER(POINTER(Link))),
			("c",POINTER(Link)),("current_iterations",c_int),("steps_on_diff_proc",c_int),("iters_removed",c_int),
			("distance",c_uint),("rejected",c_int),("save_flag",c_ushort),("peak_flag",c_ushort),
			("qvs",POINTER(QVSData)),("pos_offset",c_long),
			("expected_file_vals",c_uint),("dam",c_ushort),("res",c_ushort),
			("dim",c_uint),("diff_start",c_uint),("no_ini_start",c_uint),("num_dense",c_uint),("dense_indices",POINTER(c_uint)),
			("forcing_buff",POINTER(POINTER(ForcingData))),
			("forcing_change_times",POINTER(c_double)),("forcing_values",POINTER(c_double)),("forcing_indices",POINTER(c_uint)),
			("output_user",c_void_p),("peakoutput_user",c_void_p),
			("user",c_void_p),
			("last_eta",c_double),
			("JMatrix",POINTER(MAT)),("CoefMat",POINTER(MAT)),("Z_i",POINTER(POINTER(VEC))),("sol_diff",POINTER(VEC)),
			("h_old",c_double),("value_old",c_double),("compute_J",c_short),("compute_LU",c_short),
			("state",c_int),("discont",POINTER(c_double)),("discont_count",c_uint),("discont_start",c_uint),
			("discont_end",c_uint),("discont_send_count",c_uint),("discont_send",POINTER(c_double)),("discont_order_send",POINTER(c_uint))]

ASYNCH_SETPARAMSIZES_DATATYPE = CFUNCTYPE(None,POINTER(UnivVars),c_void_p)
ASYNCH_CONVERT_DATATYPE = CFUNCTYPE(None,POINTER(VEC),c_uint,c_void_p)
ASYNCH_ROUTINES_DATATYPE = CFUNCTYPE(None,POINTER(Link),c_uint,c_uint,c_ushort,c_void_p)
ASYNCH_PRECALCULATIONS_DATATYPE = CFUNCTYPE(None,POINTER(Link),POINTER(VEC),POINTER(VEC),c_uint,c_uint,c_ushort,c_uint,c_void_p)
ASYNCH_INITIALIZEEQS_DATATYPE = CFUNCTYPE(c_int,POINTER(VEC),POINTER(VEC),POINTER(QVSData),c_ushort,POINTER(VEC),c_uint,c_uint,c_uint,c_void_p,c_void_p)

# asynchsolver class with interface functions **********************************************************

class asynchsolver:
	def __init__(self,comm,my_rank):
		#self.lib = pydll.LoadLibrary('./libs/libasynch_py.so')
		self.lib = cdll.LoadLibrary(ASYNCH_LIBRARY_LOCATION)
		self.comm = comm
		self.np = self.comm.Get_size()
		self.my_rank = my_rank
                address = MPI._addressof(comm)
                comm_ptr = c_void_p(address)
                ranks = [self.my_rank]
		self.asynch_obj = self.lib.Asynch_Init_py(comm_ptr)
		self.tempfiles_exist = False
		#ranks = numpy.zeros((self.np,1),numpy.dtype('i4'))
		#for i in range(self.np):	ranks[i] = i
		#self.asynch_obj = self.lib.Asynch_Init_py(self.np,ranks.ctypes.data)

		#Set return types of functions. Default is c_int.
		self.lib.Asynch_Parse_GBL.restype = None
		self.lib.Asynch_Load_Network.restype = None
		self.lib.Asynch_Partition_Network.restype = None
		self.lib.Asynch_Load_Network_Parameters.restype = None
		self.lib.Asynch_Load_Numerical_Error_Data.restype = None
		self.lib.Asynch_Initialize_Model.restype = None
		self.lib.Asynch_Load_Initial_Conditions.restype = None
		self.lib.Asynch_Load_Forcings.restype = None
		self.lib.Asynch_Load_Dams.restype = None
		self.lib.Asynch_Load_Save_Lists.restype = None
		self.lib.Asynch_Finalize_Network.restype = None
		self.lib.Asynch_Calculate_Step_Sizes.restype = None
		self.lib.Asynch_Free.restype = None
		self.lib.Asynch_Refresh_Forcings.restype = None
		self.lib.Asynch_Advance.restype = None
		self.lib.Asynch_Prepare_Output.restype = None
		self.lib.Asynch_Prepare_Temp_Files.restype = None
		self.lib.Asynch_Prepare_Peakflow_Output.restype = None
		self.lib.Asynch_Set_Database_Connection.restype = None
		self.lib.Asynch_Get_Total_Simulation_Time.restype = c_double
		self.lib.Asynch_Set_Total_Simulation_Time.restype = None
		self.lib.Asynch_Get_Last_Rainfall_Timestamp.restype = c_uint
		self.lib.Asynch_Set_Last_Rainfall_Timestamp.restype = None
		self.lib.Asynch_Get_First_Rainfall_Timestamp.restype = c_uint
		self.lib.Asynch_Set_First_Rainfall_Timestamp.restype = None
		self.lib.Asynch_Set_RainDB_Starttime.restype = None
		self.lib.Asynch_Set_Init_File.restype = None
		self.lib.Asynch_Get_Number_Links.restype = c_uint
		self.lib.Asynch_Get_Local_Number_Links.restype = c_uint
		self.lib.Asynch_Set_System_State_py.restype = None
		self.lib.Asynch_Reset_Peakflow_Data.restype = None
		self.lib.Asynch_Get_Local_LinkID.restype = c_uint
		self.lib.Asynch_Get_Init_Timestamp.restype = c_uint
		self.lib.Asynch_Copy_Local_OutputUser_Data.restype = None
		self.lib.Asynch_Set_Size_Local_OutputUser_Data.restype = None
	 	self.lib.Asynch_Get_Size_Global_Parameters.restype = c_uint
         	self.lib.Asynch_Create_Local_Output.restype = c_void_p
                self.lib.Asynch_Free_Local_Output.restype = None

		#Functions created specifically for the Python interface
		self.lib.C_inc_ref.restype = None
		self.lib.Allocate_CUINT_Array.restype = POINTER(c_uint)
		self.lib.Free_PythonInterface.restype = None
		self.lib.SetParamSizes_py.restype = None
		self.lib.InitRoutines_py.restype = None
		self.lib.Asynch_Copy_Local_OutputUser_Data_py.restype = None

	def __del__(self):
		if self.tempfiles_exist == True:
			self.Delete_Temporary_Files()

		self.lib.Free_PythonInterface(self.asynch_obj)
		self.lib.Asynch_Free(self.asynch_obj)

	def Custom_Model(self,SetParamSizes,Convert,Routines,Precalculations,InitializeEqs):
		return self.lib.Asynch_Custom_Model_py(self.asynch_obj,SetParamSizes,Convert,Routines,Precalculations,InitializeEqs,py_object(self.lib))

	#Routines to initialize the system
	def Parse_GBL(self,gbl_filename):
		self.lib.Asynch_Parse_GBL(self.asynch_obj,gbl_filename)

	def Load_Network(self):
		self.lib.Asynch_Load_Network(self.asynch_obj)

	def Partition_Network(self):
		self.lib.Asynch_Partition_Network(self.asynch_obj)

	def Load_Network_Parameters(self,load_all):
		if load_all == True:
			c_load_all = 1
		else:
			c_load_all = 0
		self.lib.Asynch_Load_Network_Parameters(self.asynch_obj,c_load_all)

	def Load_Numerical_Error_Data(self):
		self.lib.Asynch_Load_Numerical_Error_Data(self.asynch_obj)

	def Initialize_Model(self):
		self.lib.Asynch_Initialize_Model(self.asynch_obj)

	def Load_Initial_Conditions(self):
		self.lib.Asynch_Load_Initial_Conditions(self.asynch_obj)

	def Load_Forcings(self):
		self.lib.Asynch_Load_Forcings(self.asynch_obj)
	
        def Refresh_Forcings(self):
		self.lib.Asynch_Refresh_Forcings(self.asynch_obj)

	def Load_Dams(self):
		self.lib.Asynch_Load_Dams(self.asynch_obj)

	def Load_Save_Lists(self):
		self.lib.Asynch_Load_Save_Lists(self.asynch_obj)

	def Finalize_Network(self):
		self.lib.Asynch_Finalize_Network(self.asynch_obj)

	def Calculate_Step_Sizes(self):
		self.lib.Asynch_Calculate_Step_Sizes(self.asynch_obj)

	#Forcing routines
	def Activate_Forcing(self,idx):
		return self.lib.Asynch_Activate_Forcing(self.asynch_obj,idx)

	def Deactivate_Forcing(self,idx):
		return self.lib.Asynch_Deactivate_Forcing(self.asynch_obj,idx)

	#Advance solver
	def Advance(self,print_flag):
		if print_flag == True:
			c_print_flag = 1
		else:
			c_print_flag = 0
		self.lib.Asynch_Advance(self.asynch_obj,c_print_flag)

	#Data file routines
	def Prepare_Output(self):
		self.lib.Asynch_Prepare_Output(self.asynch_obj)

	def Prepare_Temp_Files(self):
		self.tempfiles_exist = True
		self.lib.Asynch_Prepare_Temp_Files(self.asynch_obj)

	def Prepare_Peakflow_Output(self):
		self.lib.Asynch_Prepare_Peakflow_Output(self.asynch_obj)

	def Create_Output(self,additional_out):
		return self.lib.Asynch_Create_Output(self.asynch_obj,additional_out)
	
	def Create_Local_Output(self,additional_out,str_length):
		return self.lib.Asynch_Create_Local_Output(self.asynch_obj,additional_out,str_length)
	
        def Free_Local_Output(self,ptr):
		return self.lib.Asynch_Free_Local_Output(ptr)

	def Create_Peakflows_Output(self):
		return self.lib.Asynch_Create_Peakflows_Output(self.asynch_obj)

	def Delete_Temporary_Files(self):
		self.tempfiles_exist = False
		return self.lib.Asynch_Delete_Temporary_Files(self.asynch_obj)

	def Write_Current_Step(self):
		return self.lib.Asynch_Write_Current_Step(self.asynch_obj)

	#Snapshot
	def Take_System_Snapshot(self,name):
		return self.lib.Asynch_Take_System_Snapshot(self.asynch_obj,name)

	#Set and get routines
	def Set_Database_Connection(self,database_info,conn_idx):
		self.lib.Asynch_Set_Database_Connection(self.asynch,database_info,conn_idx)

	def Get_Total_Simulation_Time(self):
		return self.lib.Asynch_Get_Total_Simulation_Time(self.asynch_obj)

	def Set_Total_Simulation_Time(self,new_time):
		self.lib.Asynch_Set_Total_Simulation_Time(self.asynch_obj,c_double(new_time));

	def Get_Last_Rainfall_Timestamp(self,forcing_idx):
		return self.lib.Asynch_Get_Last_Rainfall_Timestamp(self.asynch_obj,forcing_idx)

	def Set_Last_Rainfall_Timestamp(self,epoch_timestamp,forcing_idx):
		self.lib.Asynch_Set_Last_Rainfall_Timestamp(self.asynch_obj,epoch_timestamp,forcing_idx)

	def Get_First_Rainfall_Timestamp(self,forcing_idx):
		return self.lib.Asynch_Get_First_Rainfall_Timestamp(self.asynch_obj,forcing_idx)

	def Set_First_Rainfall_Timestamp(self,epoch_timestamp,forcing_idx):
		self.lib.Asynch_Set_First_Rainfall_Timestamp(self.asynch_obj,epoch_timestamp,forcing_idx)

	def Set_RainDB_Starttime(self,epoch_timestamp,forcing_idx):
		self.lib.Asynch_Set_RainDB_Starttime(self.asynch_obj,epoch_timestamp,forcing_idx)

	def Set_Init_File(self,filename):
		self.lib.Asynch_Set_Init_File(self.asynch_obj,filename)

	def Get_Number_Links(self):
		return self.lib.Asynch_Get_Number_Links(self.asynch_obj)

	def Get_Local_Number_Links(self):
		return self.lib.Asynch_Get_Local_Number_Links(self.asynch_obj)

	def Get_Local_LinkID(self,location):
		return self.lib.Asynch_Get_Local_LinkID(self.asynch_obj,location)

	def Set_Init_Timestamp(self,epoch_timestamp):
		return self.lib.Asynch_Set_Init_Timestamp(self.asynch_obj,epoch_timestamp)

	def Get_Init_Timestamp(self):
		return self.lib.Asynch_Get_Init_Timestamp(self.asynch_obj)

	def Get_Size_Global_Parameters(self):
		return self.lib.Asynch_Get_Size_Global_Parameters(self.asynch_obj)

	def Get_Global_Parameters(self):
		n = self.lib.Asynch_Get_Size_Global_Parameters(self.asynch_obj)
		c_array_type = (c_double*(n))
		arr = c_array_type()
		ret_val = self.lib.Asynch_Get_Global_Parameters(self.asynch_obj,arr)
		return [list(arr), ret_val]

	def Set_Global_Parameters(self,gparams):
		c_array_type = (c_double*len(gparams))
		arr = c_array_type()
		for i in range(len(gparams)):
			arr[i] = gparams[i]
		return self.lib.Asynch_Set_Global_Parameters(self.asynch_obj,arr,len(gparams))

	#Probably not the most efficient. This currently assumes every proc has space for every link.
	def Set_System_State(self,t_0,values):
		c_array_type = ( c_double * (len(values[0]) * len(values)) )
		arr = c_array_type()
		for i in range(len(values)):
			for j in range(len(values[i])):
				arr[i*len(values[i]) + j] = values[i][j]
		self.lib.Asynch_Set_System_State_py(self.asynch_obj,c_double(t_0),arr)

	def Reset_Peakflow_Data(self):
		self.lib.Asynch_Reset_Peakflow_Data(self.asynch_obj)

	def Set_Forcing_State(self,idx,t_0,first_file,last_file):
		return self.lib.Asynch_Set_Forcing_State(self.asynch_obj,idx,c_double(t_0),first_file,last_file)

	def Set_Temp_Files(self,set_time,set_value,output_idx):
		return self.lib.Asynch_Set_Temp_Files(self.asynch_obj,c_double(set_time),byref(set_value),output_idx)

	def Reset_Temp_Files(self,set_time):
		return self.lib.Asynch_Reset_Temp_Files(self.asynch_obj,c_double(set_time))

	def Get_Peakflow_Output_Name(self):
		peakflowname = 1024*'\0'
		ret_value = self.lib.Asynch_Get_Peakflow_Output_Name(self.asynch_obj,c_char_p(peakflowname))
		i = 0
		for i in range(0,1024):
			if peakflowname[i] == '\0':	break
		true_peakflowname = peakflowname[0:i]
		return [true_peakflowname, ret_value]

	def Set_Peakflow_Output_Name(self,peakflowname):
		return self.lib.Asynch_Set_Peakflow_Output_Name(self.asynch_obj,c_char_p(peakflowname))

	def Get_Snapshot_Output_Name(self):
		filename = 1024*'\0'
		ret_value = self.lib.Asynch_Get_Snapshot_Output_Name(self.asynch_obj,filename)
		i = 0
		for i in range(0,1024):
			if filename[i] == '\0':	break
		true_filename = filename[0:i]
		return [true_filename, ret_value]

	def Set_Snapshot_Output_Name(self,filename):
		return self.lib.Asynch_Get_Snapshot_Output_Name(asynch,filename)

	#Routines for output
	def Set_Output(self,name,data_type,func,used_states_list):
		#Note: This only allows Python functions for the output.
		if used_states_list == None:
			num_states = 0
		else:
			num_states = len(used_states_list)
		used_states = numpy.array(used_states_list)
		return self.lib.Asynch_Set_Output(self.asynch_obj,name,data_type,cast(func,ASYNCH_OUTPUT_DATATYPE),used_states.ctypes.data,num_states)

	def Check_Output(self,name):
		return self.lib.Asynch_Check_Output(self.asynch_obj,name)

	def Check_Peakflow_Output(self,name):
		return self.lib.Asynch_Check_Peakflow_Output(self.asynch_obj,name)

	def Set_Peakflow_Output(self,name,func):
		return self.lib.Asynch_Set_Peakflow_Output(self.asynch_obj,name,func)

	def Create_OutputUser_Data(self,data_size):
		return self.lib.Asynch_Create_OutputUser_Data(self.asynch_obj,data_size)

	def Free_OutputUser_Data(self):
		return self.lib.Asynch_Free_OutputUser_Data(self.asynch_obj)

	def Copy_Local_OutputUser_Data(self,location,source):
		self.lib.Asynch_Copy_Local_OutputUser_Data.restype = None
		self.lib.Asynch_Copy_Local_OutputUser_Data_py(self.asynch_obj,location,py_object(source),sys.getsizeof(source))

	def Set_Size_Local_OutputUser_Data(self,location,size):
		self.lib.Asynch_Set_Size_Local_OutputUser_Data.restype = None
		self.lib.Asynch_Set_Size_Local_OutputUser_Data(self.asynch_obj,location,size)

class Assim:
    def __init__(self,filename):
        self.first_time = True
        self.asynch_data = self.get_asynch_params(filename)
        self.current_time = self.asynch_data['time_window'][0]
        (self.full_comm, self.local_comm, self.my_rank, self.nproc, self.my_ens_num, self.ens_list) = self.make_comm()
        self.asynch_dict = self.make_asynch_dict()
        self.init_cond, self.init_id_list = self.make_init_cond()
        self.global_params = self.make_global_params()
        self.link_vars = len(self.asynch_data['id_list'])*self.asynch_data['link_var_num']
        self.num_steps = self.asynch_data['num_steps']
        if self.my_rank == 0:
            self.forcings = self.get_forcings() 

    def make_comm(self):
        cfull = MPI.COMM_WORLD
        nproc = cfull.Get_size()
        my_rank = cfull.Get_rank()
        color = my_rank
        key = 0
        csmall = cfull.Split(color,key)
        ens_num = self.asynch_data['assim_params']['ens_num']
        my_ens_num = int(np.ceil(float(ens_num)/float(nproc)))
        ens_list = np.array([i for i in range(my_rank*my_ens_num,(my_rank+1)*my_ens_num)])
        return (cfull, csmall, my_rank, nproc, my_ens_num, ens_list) 

    def make_init_cond(self):
        (ic,init_id_list) = read_ini(self.asynch_data['init_cond'])
        init_cond = np.zeros((len(ic),self.my_ens_num))
        for n in range(self.my_ens_num):
            init_cond[:,n] = ic 
        init_cond = self.perturb(init_cond,initial=True,types='var')
        return init_cond, init_id_list   
 
    def make_global_params(self):
        num_param = len(self.asynch_data['init_global_params'])
        global_params = np.zeros((num_param,self.my_ens_num))
        for n in range(self.my_ens_num):
            global_params[:,n] = self.asynch_data['init_global_params']
        global_params = self.perturb(global_params,initial=True,types='params')
        return global_params

    def free_asynch_dict(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        for n in self.ens_list:
            if n < ens_num:
                del self.asynch_dict[n]

    def free_asynch_dict_element(self,n):
        ens_num = self.asynch_data['assim_params']['ens_num']
        if n < ens_num:
            del self.asynch_dict[n]

    def make_asynch_dict_element(self,n):
        ens_num = self.asynch_data['assim_params']['ens_num']
        if n < ens_num:
            self.asynch_dict[n] = asynchsolver(self.local_comm,0)

    def make_asynch_dict(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        asynch_dict = {}
        for n in self.ens_list:
            if n < ens_num:
                asynch_dict[n] = asynchsolver(self.local_comm,0)
        return asynch_dict

    def write_gbl(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        asynch_dict = {}
        step_size = self.asynch_data['step_size']
        num_steps = self.asynch_data['num_steps']
        for n in self.ens_list:
            if n < ens_num:
                n_idx = np.nonzero(self.ens_list==n)[0][0]
                create_gbl(n, self.global_params[:,n_idx], self.asynch_data, self.current_time, step_size, num_steps)

    def get_forcings(self):
        if self.my_rank == 0:
            forcing_data = get_forcings(self.asynch_data)
            return forcing_data

    def write_forcings(self):
        if self.my_rank == 0:
            if self.first_time == True:
                rain_start = self.current_time
            else:
                rain_start = self.current_time-60*(self.asynch_data['num_steps']-1)*self.asynch_data['step_size']
            make_forcing_files(self.forcings, self.asynch_data, rain_start)

    def write_sav(self):
        if self.my_rank == 0:
            write_sav(self.asynch_data)

    def write_ini(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        asynch_dict = {}
        link_var_num = self.asynch_data['link_var_num']
        if self.first_time == True:
            use_id_list = self.init_id_list
        else:
            use_id_list = np.array(self.asynch_data['id_list'])
        for n in self.ens_list:
            if n < ens_num:
                n_idx = np.nonzero(self.ens_list==n)[0][0]
                create_ini(self.asynch_data['tmp_folder']+str(n)+'.ini',use_id_list,link_var_num,self.asynch_data['model_num'],self.init_cond[:,n_idx])

    def get_asynch_params(self,filename):
        asynch_data = get_asynch_params(filename) 
        asynch_data['id_list'] = get_ids(asynch_data)
        return asynch_data

    def assimilate(self,state,measure):
        ens_anal = self.asynch_data['assim'].assimilate(state,self.asynch_data,measure)
        weights = self.asynch_data['assim'].weights
        return (ens_anal,weights)

    def fix_states(self,ens_anal):
        state = ens_anal[:-len(self.asynch_data['init_global_params']),:]
        params = ens_anal[-len(self.asynch_data['init_global_params']):,:]
        (state,params) = fix_states(self.asynch_data,state,params)
        ens_anal = np.vstack((state,params))
        return ens_anal

    def get_current_meas(self):
        if my_rank == 0:
            id_list = self.asynch_dict['id_list']
            link_var_num = self.asynch_dict['link_var_num']
            num_steps = self.asynch_dict['num_steps']
            step_size = self.asynch_dict['step_size']
            for dsource in self.asynch_data['measure']:
                dsource.get_current_meas(self.current_time - (num_steps-1)*step_size*60, id_list, link_var_num, num_steps, step_size)

    def advance(self,buffer_size):
        ens_num = self.asynch_data['assim_params']['ens_num']
        num_link = len(self.asynch_data['id_list'])
        link_var_num = self.asynch_data['link_var_num']
        num_steps = self.asynch_data['num_steps']
        num_param = len(self.asynch_data['init_global_params'])
        state = np.zeros(self.init_cond.shape)
        state_out = np.zeros((num_link*link_var_num*num_steps+num_param,self.my_ens_num))
        if self.first_time != True:
           self.free_asynch_dict()
           self.asynch_dict = self.make_asynch_dict()
        for n in self.ens_list:
            if n < ens_num:
                failed = 1
                while failed == 1:
                    n_idx = np.nonzero(self.ens_list==n)[0][0]
                    self.asynch_dict[n].Parse_GBL(str(self.asynch_data['tmp_folder'] + str(n) + '.gbl'))
                    self.asynch_dict[n].Load_Network()
                    self.asynch_dict[n].Partition_Network()
                    self.asynch_dict[n].Load_Network_Parameters(False)
                    self.asynch_dict[n].Load_Dams()
                    self.asynch_dict[n].Load_Numerical_Error_Data()
                    self.asynch_dict[n].Initialize_Model()
                    self.asynch_dict[n].Load_Save_Lists()
                    self.asynch_dict[n].Load_Initial_Conditions()
                    self.asynch_dict[n].Load_Forcings()
                    self.asynch_dict[n].Finalize_Network()
                    self.asynch_dict[n].Calculate_Step_Sizes()
                    self.asynch_dict[n].Prepare_Output()
                    self.asynch_dict[n].Prepare_Temp_Files()
                    self.asynch_dict[n].Advance(True)
                    ptr = self.asynch_dict[n].Create_Local_Output(None,buffer_size)
                    output_string = cast(ptr,c_char_p).value
                    self.asynch_dict[n].Free_Local_Output(cast(ptr,c_char_p))
                    self.asynch_dict[n].Free_OutputUser_Data()
                    parameters = np.expand_dims(np.array(self.asynch_dict[n].Get_Global_Parameters()[0]),1)
                    outarray = np.fromstring(output_string,sep=',')[:-1] #removes extra comma introduced in writing csv
                    self.asynch_dict[n].Free_OutputUser_Data()
                    if len(outarray) == (num_steps+1)*link_var_num*num_link:
                        outvec = np.reshape(outarray,(num_steps+1,-1))
                        outvec = outvec[1:,:].flatten()
                        failed = 0
                    elif len(outarray) == (num_steps)*link_var_num*num_link:
                        outvec = np.reshape(outarray,(num_steps,-1)).flatten()
                        failed = 0
                    else:
                        self.free_asynch_dict_element(n)
                        self.make_asynch_dict_element(n)
                        print('Incorrect size of output in run ' + str(n) + ', size='+str(len(outarray))+', retrying')
                        continue
                    outvec = np.expand_dims(outvec,1)
                    outvec = self.perturb(outvec,types='var')
                    parameters = self.perturb(parameters,types='params')
                    state_out[:,n_idx] = np.concatenate((outvec,parameters)).flatten()
        self.advance_time()
        return state_out

    def gather_outputs(self,state):
        ens_num = self.asynch_data['assim_params']['ens_num']
        sendvbuf = state.T.flatten().astype('f')
        if self.my_rank == 0:
            recvbuf = np.empty([self.nproc,sendvbuf.shape[0]],dtype='f')
        else:
            recvbuf = None
        self.full_comm.Gather(sendvbuf, recvbuf, root=0)
	if self.my_rank == 0:
            state_full_flatten = np.array(recvbuf).flatten()
            state_full = np.zeros((state.shape[0],self.my_ens_num*self.nproc))
            for num in range(self.my_ens_num*self.nproc):
                state_full[:,num] = state_full_flatten[num*state.shape[0]:(num+1)*state.shape[0]]
            state_full = state_full[:,0:ens_num]
            return state_full
        else:
            return None
          
    def scatter_outputs(self,sendvbuf):
        num_param = len(self.asynch_data['init_global_params']) 
        link_var_num = self.asynch_data['link_var_num']
        link_num = len(self.asynch_data['id_list'])
        num_steps = self.asynch_data['num_steps']
        ens_num = self.asynch_data['assim_params']['ens_num']
        recvbuf = np.empty((self.my_ens_num*(num_param+link_var_num*link_num*num_steps)))
        if self.my_rank == 0:
            state_full = np.concatenate((sendvbuf,np.zeros((sendvbuf.shape[0],self.my_ens_num*self.nproc-ens_num))),axis=1)
            sendvbuf = state_full.T.flatten()
        self.full_comm.Scatter(sendvbuf, recvbuf, root=0)
        state = np.reshape(recvbuf,(-1,self.my_ens_num),order='f')
        return state

    def perturb(self,state,initial=False,types='var'):
        if types == 'var':
            plist = self.asynch_data['var_perturb_type']
        elif types ==  'params':
            plist = self.asynch_data['param_perturb_type']
        else:
            raise TypeError('Invalid perturb type')
        for pert in plist:
            if initial==True:
                state = pert.init_perturb(state)
            elif initial==False:
                state = pert.perturb(state)
        return state
        
    def set_global_vars(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        for n in self.ens_list:
            if n < ens_num:
                n_idx = np.nonzero(self.ens_list==n)[0][0]
                self.asynch_dict[n].Set_Global_Parameters(self.global_params[:,n].flatten().tolist())

    def advance_time(self):
        step_size = self.asynch_data['step_size']
        num_steps = self.asynch_data['num_steps']
        if self.first_time == True:
            self.current_time += 60*step_size*num_steps
        else:
            self.current_time += 60*step_size

    def write_outputs(self,state,weights):
        if self.my_rank == 0:
            step_size = self.asynch_data['step_size']
            num_steps = self.asynch_data['num_steps']
            link_var_num = self.asynch_data['link_var_num']
            id_list = self.asynch_data['id_list']
            num_links = len(id_list)
            num_param = len(self.asynch_data['init_global_params'])
            if self.first_time == True:
                self.first_time = False
                time = [self.current_time - ((num_steps-1)-tt)*60*step_size for tt in range(num_steps)]
                data = state
            else:
                time = [self.current_time]
                data = state[-(num_param+num_links*link_var_num):,:]
            write_results(data,id_list,link_var_num,time,self.asynch_data,weights)


    def get_current_meas(self):
        if self.my_rank == 0:
            num_steps = self.asynch_data['num_steps']
            step_size = self.asynch_data['step_size']
            id_list = self.asynch_data['id_list']
            link_var_num = self.asynch_data['link_var_num']
            start_time = self.current_time-60*step_size*(num_steps-1)
            for dsource in self.asynch_data['measure']:
                dsource.get_current_meas(start_time, id_list, link_var_num, num_steps, step_size)

    def get_meas(self):
        if self.my_rank == 0:
            for dsource in self.asynch_data['measure']:
                dsource.get_meas(self.asynch_data['time_window']) 

    def create_output_folders(self):
        if self.my_rank == 0:
            create_output_folders(self.asynch_data)

    def create_tmp_folders(self):
        ens_num = self.asynch_data['assim_params']['ens_num']
        for n in self.ens_list:
            if n < ens_num:
                create_tmp_folder(self.asynch_data,n)

    def remove_files(self):
        for s in self.ens_list:
            self.asynch_dict[s].Delete_Temporary_Files()
        if self.my_rank == 0:
            remove_files(self.asynch_data['tmp_folder'])
