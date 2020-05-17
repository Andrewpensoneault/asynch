import json
from assimilation import particle, enkf, no_assimilate

def read_json(filename):
   """Reads json files"""
    with open('filename') as f:
        data = json.load(f)
    return data

def process_asynch_params(data):
    assim_params = data['assim_params']
    measure_params = data['measure_params']    
    var_perturb_params = data['var_perturb_params'] 
    param_perturb_params = data['param_perturb_params'] 

    if data['assim_type'] == 'particle':
    	assim = particle(assim_params)
    elif data['assim_type'] == 'enkf':
        assim = enkf(assim_params)
    elif data['assim_type'] == 'no_assimilate':
        assim = no_assimilate(assim_params)
    else:
        raise ValueError('Invalid assim_type')

    for i in range(len(data['measure_type'])):
        if data['measure_type'][i] == 'links':
            assim = link(measure_params[i])
        elif data['measure_type'][i] == 'soil_moisture':
            assim = soil_moisture(measure_params[i])
        else:
            raise ValueError('Invalid measure_type')

    for i in range(len(data['var_perturb_type'])):
        if data['perturb_type'][i] == 'percent':
            assim = percent(assim_params[i])
        elif data['perturb_type'][i] == 'absolute':
            assim = absolute(assim_params[i])
        elif data['perturb_type'][i] == 'per_variable_percent':
            assim = variable_percent(assim_params[i]) 
        elif data['perturb_type'][i] == 'per_variable_absolute':
            assim = variable_absolute(assim_params[i]) 
        else:
            raise ValueError('Invalid var_perturb_type')


    for i in range(len(data['param_perturb_type'])):
        if data['perturb_type'][i] == 'percent':
            assim = percent(assim_params[i])
        elif data['perturb_type'][i] == 'absolute':
            assim = absolute(assim_params[i])
        elif data['perturb_type'][i] == 'per_variable_percent':
            assim = variable_percent(assim_params[i]) 
        elif data['perturb_type'][i] == 'per_variable_absolute':
            assim = variable_absolute(assim_params[i]) 
        else:
            raise ValueError('Invalid param_perturb_type')

def get_model_params(data):
    """ File which reads initial condition to get model parameters (only ini for now, will expand)"""
    ini_filename = data['init_cond']
    with open(ini_filename, "r") as f:
       counter = 0
       while counter < 5: #the first 5 line of the ini folder contain all relevent data
          line = f.read(1)
          if not line.strip(): 
              counter += 1
              if counter == 1:
                  data['model_num'] = int(line)
              elif counter == 2:
                  data['link_num'] = int(line)
              elif counter == 5:
                  data['link_var_num'] = len(list(map(int, line.split(" ")))


def get_asynch_params(filename):
   """Reads parameter json and replaces function names with respective function"""
    data = read_json(filename)
    data = process_asynch_params(data)
    data = get_model_params(data)
    return data
