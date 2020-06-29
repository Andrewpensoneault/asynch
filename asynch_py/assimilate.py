import numpy as np 
import scipy.linalg
import scipy.stats as spstats
from asynch_py.process_error import process_error

class particle():
    def __init__(self,data):
        self.ens_num = data['ens_num']
        if data['likelihood_type'] == 'gaussian':
            neglog = lambda x,y,Rinv,H: np.diag(np.matmul(np.matmul((H(x)-y).transpose(),Rinv/2),(H(x)-y)))
            self.likelihood_type = lambda x, y, Rinv, H: np.exp(np.min(neglog(x,y,Rinv,H))-neglog(x,y,Rinv,H))
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
       if len(y)>0:
           invR = np.diag(1/np.array(R))
           y = np.expand_dims(y,1)
           H = lambda x: np.vstack([Hl(x) for Hl in Hlist])
           weight = self.weights*self.likelihood_type(state,y,invR,H)
           weight = weight/np.sum(weight)
           if (1/np.sum(weight**2)) < self.eff_sample_threshold:
               best = np.argmax(weight)
               xbest = state[:,best]
               xk = np.arange(self.ens_num)
               resample_dist = spstats.rv_discrete(values=(xk,weight.flatten()))
               choice = resample_dist.rvs(size=self.ens_num)
               state = state[:,choice]
               weight = (1./self.ens_num)*np.ones((1,self.ens_num))
               for pert in self.var_roughing_type:
                   state[:-param_num,:] = pert.perturb(state[:-param_num,:])
               for pert in self.param_roughing_type:
                   state[-param_num:,:] = pert.perturb(state[-param_num:,:])
               state[:,0] = xbest
           self.weights = weight 
       return state

class no_assimilate():
    def __init__(self,data):
        self.ens_num = data['ens_num']
        self.weights = np.ones((1,self.ens_num))/self.ens_num
    def assimilate(self,state_data,assim_data,measure):
        return state_data

class enkf():
    def __init__(self,data):
        self.ens_num = data['ens_num']
        self.weights = np.ones((1,self.ens_num))/self.ens_num
    def assimilate(self,state_data,assim_data,measure):
       param_num = len(assim_data['init_global_params'])
       R = []
       y = []
       Hlist = []
       H = lambda x: np.array([[]])
       for meas in measure:
          R.extend(np.diag(meas.R).tolist())
          Hlist += [lambda x: meas.H(x)]
          y.extend(meas.meas.flatten().tolist())
       if len(y)>0:
           invR = np.diag(1/np.array(R))
           R = np.diag(np.array(R))
           y = np.expand_dims(y,1)
           H = lambda x: np.vstack([Hl(x) for Hl in Hlist])
           Hx = H(state_data)
           HX = (Hx-np.mean(Hx,keepdims=1,axis=1))/np.sqrt(self.ens_num-1)
           X = (state_data-np.mean(state_data,keepdims=1,axis=1))/np.sqrt(self.ens_num-1)
           S = np.matmul(HX,HX.T)+R
           K = np.linalg.solve(S.T,(np.matmul(X,HX.T).T)).T
           Y = y + np.matmul(np.sqrt(R),np.random.normal(0,1,(len(y),self.ens_num)))
           state = state_data + np.matmul(K,(Y-Hx))
       else:
            state = state_data
       return state


class srenkf():
    def __init__(self,data):
        self.ens_num = data['ens_num']
        self.weights = np.ones((1,self.ens_num))/self.ens_num
    def assimilate(self,state_data,assim_data,measure):
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
       if len(y) > 0:
           invR = np.diag(1/np.array(R))
           R = np.diag(np.array(R))
           y = np.expand_dims(y,1)
           H = lambda x: np.vstack([Hl(x) for Hl in Hlist])
           Hx = H(state_data)
           HX = (Hx-np.mean(Hx,keepdims=1,axis=1))/np.sqrt(self.ens_num-1)
           X = (state_data-np.mean(state_data,keepdims=1,axis=1))/np.sqrt(self.ens_num-1)
           S = np.matmul(HX,HX.T)+R
           K = np.linalg.solve(S.T,(np.matmul(X,HX.T).T)).T
           m = np.mean(state_data,keepdims=1,axis=1) + np.matmul(K,(y-np.mean(Hx,keepdims=1,axis=1)))
           T = np.linalg.inv(np.eye(self.ens_num)+np.matmul(HX.T,np.matmul(invR,HX)))
           Xp = np.matmul(X,scipy.linalg.sqrtm(T))
           xi = np.sqrt(self.ens_num-1)*Xp
           state = np.real(m + xi)
       else:
           state = state_data
       return state
