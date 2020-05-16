import json
from assimilation import particle, enkf, no_assimilate

def read_json(filename):
   """Reads json files"""
    with open('filename') as f:
        data = json.load(f)
    return data

def read_asynch_params(filename):
   """Reads parameter json and replaces functions with respective function"""
    data = read_json(filename):

    assim_params = data['assim_params']
    measure_params = data['measure_params']    
    perturb_params = data['perturb_params'] 

    if data['assim_type'] == 'particle':
    	assim = particle(assim_params)
    elif data['assim_type'] == 'enkf':
        assim = enkf(assim_params)
    elif data['assim_type'] == 'no_assimilate':
        assim = no_assimilate(assim_params)

    if data['measure_type'] == 'links':
    	assim = link(measure_params)

   for i in range(len(data['perturb_type'])
       if data['perturb_type'][i] == 'percent':
           assim = percent(assim_params)
       elif data['perturb_type'][i] == 'absolute':
           assim = absolute(assim_params)
       elif data['perturb_type'][i] == 'variable_percent':
           assim = variable_percent(assim_params) 
       elif data['perturb_type'][i] == 'variable_absolute':
           assim = variable_absolute(assim_params) 
        
