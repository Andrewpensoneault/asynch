from asynch_py.perturb import absolute, percent, per_variable_absolute, per_variable_percent

def process_error(data,type_name,out_name,param_name):
    for i in range(len(data[type_name])):
        if data[type_name][i] == 'percent':
            data[out_name][i] = percent(data[param_name][i])
        elif data[type_name][i] == 'absolute':
            data[out_name][i] = absolute(data[param_name][i])
        elif data[type_name][i] == 'per_variable_percent':
            data[type_name][i] = per_variable_percent(data[param_name][i]) 
        elif data[type_name][i] == 'per_variable_absolute':
            data[out_name][i] = per_variable_absolute(data[param_name][i]) 
        else:
            raise ValueError('Invalid var_perturb_type')
