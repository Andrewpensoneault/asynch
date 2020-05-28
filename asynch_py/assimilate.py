import numpy as np 
import scipy.stats as spstats
from asynch_py.process_error import process_error

class particle():
    def __init__(self,data):
        self.ens_num = data['ens_num']
        if data['likelihood_type'] == 'gaussian':
            self.likelihood_type = lambda x, y, Rinv, H: np.exp(np.max(np.diag(np.matmul(np.matmul((H(x)-y).transpose(),(Rinv/2)),(H(x)-y))))-np.diag(np.matmul(np.matmul((H(x)-y).transpose(),(Rinv/2)),(H(x)-y))))
        self.eff_sample_threshold = data['eff_samp_threshold']
        process_error(data,'var_roughing_type','var_roughing_type','var_roughing_params')
        process_error(data,'param_roughing_type','param_roughing_type','param_roughing_params')
        self.var_roughing_type = data['var_roughing_type']
        self.param_roughing_type = data['param_roughing_type']
        self.weights = np.ones((1,self.ens_num))/self.ens_num
    
    def assimilate(self,state,assim_data,measure):
       param_num = len(assim_data['init_global_params'])
       R = []
       y = []
       Hlist = []
       i = 0
       H = lambda x: np.array([[]])
       for meas in measure:
          R.extend(np.diag(meas.R).tolist())  
          Hlist += [lambda x: meas.H(x)]
          y.extend(meas.meas.flatten().tolist())
       invR = np.diag(1/np.array(R))
       y = np.expand_dims(y,1)
       H = lambda x: np.vstack([Hl(x) for Hl in Hlist])
       weight = self.weights*self.likelihood_type(state,y,invR,H)
       weight = weight/np.sum(weight)
       if (1/np.sum(weight**2)) < self.eff_sample_threshold:
           xk = np.arange(self.ens_num)
           resample_dist = spstats.rv_discrete(values=(xk,weight.flatten()))
           choice = resample_dist.rvs(size=self.ens_num)
           state = state[:,choice]
           weight = (1/self.ens_num)*np.ones((1,self.ens_num))
           for pert in self.var_roughing_type:
               state[:-param_num,:] = pert.perturb(state[:-param_num,:])
           for pert in self.param_roughing_type:
               state[-param_num:,:] = pert.perturb(state[-param_num:,:])
       self.weights = weight 
       return state

class no_assimilate():
    def __init__(self,data):
        pass

class enkf():
    def __init__(self,data):
        pass
