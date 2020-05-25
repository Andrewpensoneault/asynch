import numpy as np
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
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.error[i]*np.random.normal(0,1,1))
        return tmp

    def init_perturb(self,ens):
        tmp = np.ones(ens.shape)
        for i in range(self.variable_num):
            tmp[i::self.variable_num,:] = ens[i::self.variable_num,:]*(1+self.init_error[i]*np.random.normal(0,1,1))
        return tmp
    
    def get_var(self,index,ens):
        var = (ens*self.init_error[index])**2
        return var

class per_variable_absolute():
    def __init__(self,data):
        pass
