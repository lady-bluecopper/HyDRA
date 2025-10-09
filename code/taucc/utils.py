import os
import logging as l
import datetime


def CreateOutputFile(partial_name, own_directory=False, date=True, overwrite=False):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
    overwrite: boolean. Default: False.
        If true, overwrite the existent file (if there exists a file with the same name)
        If false, append the new results.
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - model: 'CoClust' or 'CC'
        - dim_x: dimension of the tensor on mode 0
        - dim_y: dimension of the tensor on mode 1
        - x_num_classes: correct number of clusters on mode 0
        - y_num_classes: correct number of clusters on mode 1
        - noise: only for synthetic tensors. Amount of noise added to the perfect tensor
        - tau_x: final tau_{x|y}
        - tau_y: final tau_{y|x}
        - nmi_x: normalized mutual information score on mode 0
        - nmi_y: normalized mutual information score on mode 1
        - ari_x: adjusted rand index on mode 0
        - ari_y: adjusted rand index on mode 1
        - x_num_clusters: number of clusters on mode 0 detected by CoClust
        - y_num_clusters: number of clusters on mode 1 detected by CoClust
        - execution time
        - iter: total number of iterations
        - init_clusters_x: number of initial clusters on mode 0
        - init_clusters_y: number of initial clusters on mode 1

        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    '''
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
    f = open(data_path + file_name, "a",1)
    if new:
        f.write("model,dim_x,dim_y,x_num_classes,y_num_classes,noise,tau_x,tau_y,nmi_x,nmi_y,ari_x,ari_y,x_num_clusters,y_num_clusters,execution_time,iter, init_clusters_x,init_clusters_y,n_iterations\n")
    return f, dt


def CreateLogger(input_level='INFO'):
    level = {
        'DEBUG': l.DEBUG,
        'INFO': l.INFO,
        'WARNING': l.WARNING,
        'ERROR': l.ERROR,
        'CRITICAL': l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])

    return logger
