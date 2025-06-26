
import numpy as np
from scipy.interpolate import interp1d
import os
from loguru import logger

TIMELIMIT = 1000
x_interpolated = np.linspace(0, TIMELIMIT, 5000)


TASK = 'MIS'
directory = './result'
methods = [method for method in os.listdir(directory)]

BKS = [float('inf')] * 10
for method in (os.listdir(directory)):
    instances = [instance for instance in os.listdir(f'{directory}/{method}/{TASK}')]
    for id, instance in enumerate(instances):
        with open(f'./result/{method}/{TASK}/{instance}', 'r') as f:
            a = f.read()
        a = list(eval(a))
        BKS[id] = min(BKS[id], float(a[1][-1]))


for idx, method in enumerate(os.listdir(directory)):
    instances = [instance for instance in os.listdir(f'{directory}/{method}/{TASK}')]
    pbs = []
    pgs = []
    for id, instance in enumerate(instances):
        ### BKS of this instance
        with open(f'./result/{method}/{TASK}/{instance}', 'r') as f:
            a = f.read()
        a = list(eval(a))
        a[0] = [t for t in a[0] if t <= TIMELIMIT]
        a[1] = a[1][:len(a[0])]
        a[0].append(1000)
        a[1].append(a[1][-1])


        interpolated_func = interp1d([0]+a[0], [a[1][0]]+a[1], kind='previous')
        y_interpolated = interpolated_func(x_interpolated)
        pbs.append(y_interpolated)
        pgs.append([max(0, y - BKS[id]) / abs(BKS[id]) for y in y_interpolated])

    pbs = np.array(pbs)
    pgs = np.array(pgs)
    mean_pb = np.mean(pbs, axis=0)
    std_pb = np.std(pbs, axis=0)
    mean_pg = np.mean(pgs, axis=0)
    std_pg = np.std(pgs, axis=0)
    pi = [np.trapz(y, x_interpolated) for y in pgs]
    mean_pi = np.mean(pi)
    logger.info(f'{method}: PG--{mean_pg[-1]}, PI--{mean_pi}')





