import numpy as np
import pandas as pd
from asynch_py.process_error import process_error

class links():
    def __init__(self,data): 
        data['meas_error'] = [0]*len(data['meas_error_type'])         
        process_error(data,'meas_error_type','meas_error','meas_error_params')

        self.meas_ids = np.array(data['meas_ids'])
        self.meas_error = data['meas_error']
        self.meas_error_params = data['meas_error_params']
        self.meas_folder = data['meas_folder']
        self.data = {}


    def __read_csv(self, filename, skiprow):
        contents = np.array(pd.read_csv(filename, delimiter=",",skiprows = skiprow).values)
        return contents
    
    def get_error(self, index, data):
        ind_idx = np.nonzero(self.meas_ids == index)[0][0]
        var = 0
        for pert in self.meas_error:
            var += pert.get_var(ind_idx,data) 
        return var

    def get_meas(self,time_window):
        start_time = time_window[0]
        stop_time = time_window[1]
        header_row = 1
        for ids in self.meas_ids:
            filename = self.meas_folder + str(ids) + '_meas.csv'
            contents = self.__read_csv(filename,header_row)
            times = contents[:,0]
            use_times = np.nonzero(np.logical_and(times>=start_time,times<=stop_time))[0]
            self.data[ids] = contents[use_times,:]

    def get_current_meas(self, time_start, id_list, link_var_num, num_step, step_size):
        id_num = len(id_list)
        meas = []
        idxs = []
        R = []
        for ids in self.meas_ids:
            data = self.data[ids]
            idx = np.nonzero(np.array(id_list)==ids)[0][0]
            for i in range(num_step):
                time_now = time_start + step_size*i*60
                time_next = time_start + step_size*(i+1)*60
                pot_meas = data[np.nonzero(data[:,0]<time_next)[0],:]
                pot_meas = pot_meas[np.nonzero(pot_meas[:,0]>=time_now)[0],:]
                if len(pot_meas)!=0:
                    curr_meas = pot_meas[0,1]
                    curr_idx = idx*link_var_num+link_var_num*id_num*i
                    meas += [curr_meas]
                    idxs += [curr_idx]  
                    R += [self.get_error(ids,curr_meas)]
        H = lambda x: x[idxs,:]
        meas = np.expand_dims(meas,1)
        R = np.diag(R)

        self.H = H
        self.meas = meas
        self.R = R  

class soil_moisture():
    def __init__(self,data):
        pass
