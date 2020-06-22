import numpy as np


def fix_states(asynch_data,state,params):
    lower_bounds_var = np.array([asynch_data['bounds_var'][i][0] for i in range(len(asynch_data['bounds_var']))])
    upper_bounds_var = np.array([asynch_data['bounds_var'][i][1] for i in range(len(asynch_data['bounds_var']))])
     
    lower_bounds_param = np.array([asynch_data['bounds_params'][i][0] for i in range(len(asynch_data['bounds_params']))])
    upper_bounds_param = np.array([asynch_data['bounds_params'][i][1] for i in range(len(asynch_data['bounds_params']))])

    num_var = len(lower_bounds_var)   
    num_param = len(lower_bounds_param)   

    for i in range(num_var):
        state[i::num_var,:] = np.minimum(np.maximum(state[i::num_var,:],lower_bounds_var[i]),upper_bounds_var[i])

    for i in range(num_param):
        params[i,:] = np.minimum(np.maximum(params[i,:],lower_bounds_param[i]),upper_bounds_param[i])

    return (state,params)


class absolute():
    def __init__(self,data):
        self.error = data['absolute_error']
        self.init_error = data['initial_absolute_error']
        self.variable_num = len(self.error)

    def perturb(self,ens): 
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            for j in range(ens.shape[0]//self.variable_num):
                tmp[self.variable_num*j+i,:] = ens[self.variable_num*j+i,:]+self.error[i]*np.random.normal(0,1,ens[self.variable_num*j+i,:].shape)
        return tmp

    def init_perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            for j in range(ens.shape[0]//self.variable_num):
                tmp[self.variable_num*j+i,:] = ens[self.variable_num*j+i,:]+self.init_error[i]*np.random.normal(0,1,ens[self.variable_num*j+i,:].shape)
        return tmp

    def get_var(self,index,ens):
        var = self.error[index]**2
        return var

class percent():
    def __init__(self,data):
        self.error = data['percent_error']
        self.init_error = data['initial_percent_error']
        self.variable_num = len(self.error)

    def perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.error[i]*np.random.normal(0,1,ens[i::self.variable_num,:].shape))
        return tmp

    def init_perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.init_error[i]*np.random.normal(0,1,ens[i::self.variable_num,:].shape))
        return tmp
    
    def get_var(self,index,ens):
        var = (ens*self.init_error[index])**2
        return var


class per_variable_percent():
    def __init__(self,data):
        self.error = data['percent_error']
        self.init_error = data['initial_percent_error']
        self.variable_num = len(self.error)

    def perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.error[i]*np.random.normal(0,1,(1,tmp.shape[1])))
        return tmp

    def init_perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.init_error[i]*np.random.normal(0,1,(1,tmp.shape[1])))
        return tmp
    
    def get_var(self,index,ens):
        var = (ens*self.init_error[index])**2
        return var

class per_variable_absolute():
    def __init__(self,data):
        pass
