import numpy as np
from asynch_py.assimilate import particle, enkf, no_assimilate, srenkf
from asynch_py.measure import links, soil_moisture
from asynch_py.perturb import absolute, percent, per_variable_absolute, per_variable_percent
from asynch_py.process_error import process_error

def process_asynch_params(data):
    assim_params = data['assim_params']
    measure_params = data['measure_params']    
    var_perturb_params = data['var_perturb_params'] 
    param_perturb_params = data['param_perturb_params'] 

    if data['assim_type'] == 'particle':
    	data['assim'] = particle(assim_params)
    elif data['assim_type'] == 'srenkf':
        data['assim'] = srenkf(assim_params)
    elif data['assim_type'] == 'enkf':
        data['assim'] = enkf(assim_params)
    elif data['assim_type'] == 'no_assimilate':
        data['assim'] = no_assimilate(assim_params)
    else:
        raise ValueError('Invalid assim_type')

    data['measure'] = [0]*len(data['measure_type'])
    for i in range(len(data['measure_type'])):
        if data['measure_type'][i] == 'links':
            data['measure'][i] = links(measure_params[i])
        elif data['measure_type'][i] == 'soil_moisture':
            data['measure'][i] = soil_moisture(measure_params[i])
        else:
            raise ValueError('Invalid measure_type')

    data['var_perturb'] = [0]*len(data['var_perturb_type'])
    process_error(data,'var_perturb_type','var_perturb_type','var_perturb_params')
    
    data['param_perturb'] = [0]*len(data['param_perturb_type'])
    process_error(data,'param_perturb_type','param_perturb_type','param_perturb_params')
    
    return data
