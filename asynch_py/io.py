import json
from asynch_py.process import  process_asynch_params
import numpy as np
import pandas as pd
import os
from shutil import copyfile, rmtree
import datetime

def create_tmp_folder(data,n):
    try:
        os.makedirs(data['tmp_folder']+str(n)+'/')
    except OSError:
        if not os.path.isdir(data['tmp_folder']+str(n)+'/'):
            raise

def create_output_folders(data):
    """Write output and tmp folders"""
    output_folder = data['output_folder']
    
    try:
        os.makedirs(data['tmp_folder'])
    except OSError:
        if not os.path.isdir(data['tmp_folder']):
            raise
        else:
            rmtree(data['tmp_folder'])
            os.makedirs(data['tmp_folder'])

    try:
        os.makedirs(output_folder)
    except OSError:
        if not os.path.isdir(output_folder):
            raise
        else:
            rmtree(output_folder)
            os.makedirs(output_folder)

    try:
        os.makedirs(output_folder + 'model')
    except OSError:
        if not os.path.isdir(output_folder + 'model'):
            raise

    try:
        os.makedirs(output_folder + 'measure')
    except OSError:
        if not os.path.isdir(output_folder + 'measure'):
            raise

    try:
        os.makedirs(output_folder + 'rainfall')
    except OSError:
        if not os.path.isdir(output_folder + 'rainfall'):
            raise

def get_ids(data):
    with open(data['rvr_filename']) as f:
        lines = [line.rstrip('\n')  for line in f if line != '']
    lines = list(filter(None, lines))
    lines = list(map(int,lines[1::2]))
    return lines

def write_sav(data):
    filename = data['tmp_folder'] + 'asynch.sav'
    lines = get_ids(data)
    with open(filename, 'w') as filehandle:
        for l in lines:
            filehandle.write('%d\n' % l)
    return lines

def read_json(filename):
    """Reads json files"""
    with open(filename) as f:
        data = json.load(f)
    return data

def read_csv(filename, skiprow):
    contents = np.array(pd.read_csv(filename, delimiter=",",skiprows = skiprow).values)
    return contents

def read_text_file(filename): 
    with open(filename) as f:
        lines = f.readlines() #removes all blank lines and lines which start with #
        lines = [x for y in lines for x in y.split("\n")]
        lines = [x.strip() for x in lines]
        lines = [x for x in lines if x != '']
        lines = [x for x in lines if '#' not in x]
    return lines


def write_gbl_file(filename,gbl,): 
    with open(filename) as f:
        lines = [x for x in lines if '#' not in x]
    return lines

def get_model_params(data):
    """ File which reads initial condition to get model parameters (only ini for now, will expand)"""
    ini_filename = data['init_cond']
    with open(ini_filename, "r") as f:
       counter = 0
       while counter < 5: #the first 5 line of the ini folder contain all relevent data
          line = f.readline()
          if line.strip(): 
              counter += 1
              if counter == 1:
                  data['model_num'] = int(line)
              elif counter == 2:
                  data['link_num'] = int(line)
              elif counter == 5:
                  data['link_var_num'] = len(list(map(float, line.split(" "))))
    return data

def get_asynch_params(filename):
    """Reads parameter json and replaces function names with respective function"""
    data = read_json(filename)
    data = process_asynch_params(data)
    data = get_model_params(data)
    return data

def get_rainfall(data):
    filename = data['rainfall_source']
    time_start = data['time_window'][0]
    time_stop = data['time_window'][1]

    HEADER_ROWS = 1
    contents = read_csv(filename,HEADER_ROWS)
    rain_list = np.array(contents[:,2]).astype('float')
    rain_times = np.array(contents[:,1]).astype('int')
    rain_ids = np.array(contents[:,0]).astype('int')
    index = np.nonzero(np.logical_and(rain_times>=time_start, rain_times<=time_stop))
    rain_times = rain_times[index]
    rain_list = rain_list[index]
    rain_ids = rain_ids[index]
    
    return rain_times, rain_list, rain_ids

def get_evap(data):
    filename = data['evap_source']
    lines = np.array(read_text_file(filename))
    evap = lines.astype('float')
    return evap
   
def get_forcings(data):
    """Reads forcing files (currently csv for rainfall and mon for evap)"""
    (rain_times, rain_list, rain_ids) = get_rainfall(data)
    evap = get_evap(data)
    forcing_data = {}
    forcing_data['rain_times'] = rain_times
    forcing_data['rain_list'] = rain_list
    forcing_data['rain_ids'] = rain_ids
    forcing_data['evap'] = evap
    return forcing_data

def make_forcing_files(forcing_data, asynch_data, current_time):
    current_rain = get_current_rain(forcing_data, asynch_data, current_time)
    create_str(current_rain,asynch_data)
    create_evap(asynch_data)

def get_current_rain(forcing_data, asynch_data, current_time):
    HOUR_IN_SECONDS = 3600
    MINUTE_IN_SECONDS = 60

    id_list = asynch_data['id_list']
    total_links = len(id_list)

    num_steps = asynch_data['num_steps']
    time_start = current_time
    time_step = asynch_data['step_size']

    rain_times = forcing_data['rain_times']
    rain_ids = forcing_data['rain_ids']
    rain_values = forcing_data['rain_list']


    current_rain = np.zeros((total_links,num_steps))
    for time in range(num_steps):
        time_now = time_start + MINUTE_IN_SECONDS*time_step*time
        last_hour_time = np.floor(time_now/HOUR_IN_SECONDS)*HOUR_IN_SECONDS
        next_time = time_now + MINUTE_IN_SECONDS*time_step
        last_hour_rain_idx = np.nonzero(np.logical_and(rain_times>=last_hour_time,rain_times<=next_time))[0]
        if len(last_hour_rain_idx) == 0:
           current_rain[:,time] = 0.0
        else:
            rain_ids_current = rain_ids[last_hour_rain_idx]
            rain_values_current = rain_values[last_hour_rain_idx]
            for link in range(total_links):
                current_id = asynch_data['id_list'][link]
                rain_id_index = np.nonzero(rain_ids_current==current_id)
                rain_values_current_id = rain_values_current[rain_id_index]
                if not len(rain_values_current_id) == 0:
                    current_rain[link,time] = rain_values_current_id[0]
                else:
                    current_rain[link,time] = 0.0
    return current_rain

def create_str(current_rain, data):
    filename = data['tmp_folder'] + 'asynch.str'
    id_list = data['id_list']
    total_links = len(id_list)
    time_step = data['step_size']
    num_steps = data['num_steps']

    id_contents = np.array([id_list,num_steps*np.ones(total_links,dtype="int")]).T
    id_contents = [' '.join(map(str, x)) for x in id_contents.tolist()]
    cycle_num = np.array([i*time_step for i in range(num_steps)])
    time_rain = np.tile(cycle_num, total_links)

    rain_content = np.array([time_rain, current_rain.flatten()]).T
    rain_content = [' '.join(map(str, x)) for x in rain_content.tolist()]
    tmp_rain = []
    for j in range(num_steps):
        tmp_rain += [rain_content[j::num_steps]]
    content = [val for pair in zip(id_contents, *tmp_rain) for val in pair]
    content = "\n".join(content)
    with open(filename,'w') as f:
        f.write("%s\n"%str(total_links))
        f.write("%s\n"%content)

def create_evap(data):
    output_filename = data['tmp_folder'] + 'asynch.mon'
    filename = data['evap_source']
    copyfile(filename,output_filename)

def create_gbl(n, global_params, asynch_data, current_time, time_step, num_steps):
    rvr_filename = asynch_data['rvr_filename']
    prm_filename = asynch_data['prm_filename']
    evap_filename = asynch_data['tmp_folder'] + 'asynch.mon'
    str_filename = asynch_data['tmp_folder'] + 'asynch.str'
    ini_filename = asynch_data['tmp_folder'] + str(n) + '.ini'
    tmp_filename =  asynch_data['tmp_folder'] + str(n) + '/' + str(n)
    gbl_filename = asynch_data['tmp_folder'] + str(n) + '.gbl'
    model_num = asynch_data['model_num']

    contents = read_text_file(asynch_data['gbl_sample_filename']) 
    contents[1] = str(model_num) + ' ' + str(time_step*num_steps)
    current_global = ' '.join(map(str,global_params))
    contents[13] = str(len(global_params)) + ' ' + current_global
    contents[19] = '0 ' + rvr_filename
    contents[21] = '0 ' + prm_filename
    contents[23] = '0 ' + ini_filename
    contents[27] = '1 ' + str_filename 
    contents[29] = '7 ' + evap_filename 
    contents[30] = str(asynch_data['time_window'][0]) + ' ' +  str(asynch_data['time_window'][1]) 
    contents[37] = '2 ' + str(asynch_data['step_size']) + ' ' + asynch_data['tmp_folder'] + 'asynch.csv' 
    contents[43] = '1 ' + asynch_data['tmp_folder'] + 'asynch.sav' 
    contents[47] = tmp_filename
    contents[-1] = '#' 
    
    with open(gbl_filename,'w') as f:
        for items in contents:
            f.write("%s\n" % items)

def read_ini(filename):
    lines = read_text_file(filename)
    id_list = np.array(lines[3::2])
    init_cond = lines[4::2]

    init_cond = [x.split() for x in init_cond]
    init_cond = np.array(init_cond)
    init_cond = init_cond.astype(np.float)
    id_list = id_list.astype(np.float)
    return init_cond.flatten(), id_list

def create_ini(filename,id_list,link_variable_num,model_num,state):
    total_links = len(id_list)
    var_matrix = np.zeros((total_links,link_variable_num))
    content_id = [str(int(x)) for x in id_list.tolist()]
    for j in range(link_variable_num):
        var_matrix[:,j] = state[j::link_variable_num]
    content_var = [' '.join(map(str, x)) for x in var_matrix.tolist()]
    content = [val for pair in zip(content_id, content_var) for val in pair]
    content = "\n".join(content)
    with open(filename,'w') as f:
        f.write("%s\n"%str(model_num))
        f.write("%s\n"%str(total_links))
        f.write("%s\n"%str(0))
        f.write("%s\n"% content)

def remove_files(tmp_folder):
    rmtree(tmp_folder)

def get_std(data, weight = None):
    if np.all(weight) == None:
        weight = np.ones((1,data.shape[1]))/(data.shape[1])
    mean = get_mean(data,weight)
    return (np.sqrt(data.shape[1])/np.sqrt(data.shape[1]-1))*np.sqrt(np.sum((data-mean)**2*weight,axis=1))

def get_mean(data, weight = None):
    if np.all(weight) == None:
        weight = np.ones((1,data.shape[1]))/(data.shape[1])
    return np.sum(data*weight,axis=1,keepdims=1)

def write_results(data, link_ids, link_var_num, times, asynch_data, weight=None):
    TITLES =  np.array([['dt', 'Q']])
    std = get_std(data,weight)
    mean = get_mean(data,weight)
    
    model_folder = asynch_data['output_folder'] + 'model/'
    write_ids = asynch_data['write_ids']
    
    times = np.expand_dims([datetime.datetime.utcfromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S') for t in times],axis=1)
    
    for ids in write_ids:
        p_filename = model_folder + '/' + str(ids) + '_p.csv'
        n_filename = model_folder + '/' + str(ids) + '.csv'
        m_filename = model_folder + '/' + str(ids) + '_m.csv'
        
        id_idx = np.nonzero(np.array(link_ids) == ids)[0][0]
        for t in range(len(times)):
            idx = id_idx*link_var_num + link_var_num*len(link_ids)*t
            mean_current = mean[idx]
            std_current = std[idx]
            
            p_out = np.expand_dims(['%.4f' % (mean_current + std_current)],axis=1)
            out = np.expand_dims(['%.4f' % (mean_current)],axis=1)
            m_out = np.expand_dims(['%.4f' %  np.maximum(mean_current - std_current,0)],axis=1)

            p_out = np.hstack((np.expand_dims(times[t],1),p_out))
            out = np.hstack((np.expand_dims(times[t],1),out))
            m_out = np.hstack((np.expand_dims(times[t],1),m_out))

            try:
                os.stat(p_filename)
            except OSError:
                with open(p_filename,'a') as f:
                    np.savetxt(f,TITLES, fmt="%s,%s", delimiter=",")
            with open(p_filename,'a') as f:
                np.savetxt(f, p_out, fmt="%s,%s", delimiter=",")

            try:
                os.stat(n_filename)
            except OSError:
                with open(n_filename,'a') as f:
                    np.savetxt(f,TITLES, fmt="%s,%s", delimiter=",")
            with open(n_filename,'a') as f:
                np.savetxt(f, out, fmt="%s,%s", delimiter=",")

            try:
                os.stat(m_filename)
            except OSError:
                with open(m_filename,'a') as f:
                    np.savetxt(f,TITLES, fmt="%s,%s", delimiter=",")
            with open(m_filename,'a') as f:
                np.savetxt(f, m_out, fmt="%s,%s", delimiter=",")
     

