import os
import sys
import argparse
import pathlib
import numpy as np
import random
#import pyscipopt as scp
import torch
import torch.nn as nn
import pickle
import time
from scipy import spatial
import gurobipy as gp
from gurobipy import GRB
from torch.nn.functional import softmax
import torch.nn.functional as F
#device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False

    # os.environ['PYTHONHASHSEED'] = str(seed)


    
def state_vnode_represent(v_nodes, cur_sol_val):
    epsilon = 1e-6
    return torch.cat((v_nodes,torch.tensor(cur_sol_val).unsqueeze(1)),dim=1)


def choose(weight,k):
    from scipy.special import softmax

    n = len(weight)
    lst = [i for i in range(n)]
    weights = [w+1/len(weight) for w in weight]
    prob = weights/sum(weights)
    sampled = np.random.choice(lst, size=k, replace=False, p=prob)
    fixation = list(set(lst)-set(sampled))

    # a = [0 for i in range(n)]
    # for s in sampled:
    #     a[s] = 1
    return fixation,sampled


def perturb(lst0, percent, int_idx):
    percent = min(1, percent)
    lst = lst0.copy()
    import math
    n = len(lst)

    k = math.floor(sum(lst)*percent)
    ones_indices = [i for i in range(n) if lst[i] == 1]
    zeros_indices = [i for i in range(n) if lst[i] == 0 and int_idx[i] == 1]
    k = math.ceil(min(len(ones_indices)*percent,len(zeros_indices)*percent))

    modified_ones = random.sample(ones_indices, k)
    for i in modified_ones:
        lst[i] = 0

    # 随机修改取值为 0 的元素
    modified_zeros = random.sample(zeros_indices, k)
    for i in modified_zeros:
        lst[i] = 1
    return lst



def get_a_new2(ins_name):
    epsilon = 1e-6
    import pyscipopt as scp
    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = scp.Model()

    m.setParam('parallel/maxnthreads', 1)
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    v_nodes = []
    ori_start = 6


    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[5] = 1

        v_nodes.append(tp)


    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_coeff = [0]*len(mvars)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0,1,0]
    for e in obj:
        vnm = e.vartuple[0].name  #变量名
        v = obj[e]  #目标函数系数c
        v_indx = v_map[vnm]
        obj_coeff[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            #values_spr.append(v)
            values_spr.append(1)
        v_nodes[v_indx][0] = v

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]

    obj_coeff = np.array(obj_coeff)
    obj_node[5] = np.linalg.norm(obj_coeff)

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)

    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    c_nodes.append(obj_node)
    for ind, c in enumerate(cons):
        cind = ind+1
        coeff = m.getValsLinear(c)
        dot = sum([coeff[name]*obj_coeff[v_map[name]]for name in coeff.keys()])
        norm = np.sqrt(sum([coeff[name]**2 for name in coeff.keys()]))
        cosine = dot/ norm / obj_node[-1]


        rhs = m.getRhs(c)
        lhs = m.getLhs(c)

        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:

            v_indx = v_map[k]
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(coeff[k])

            v_nodes[v_indx][2] += 1

            v_nodes[v_indx][1] += coeff[k] / lcons

            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            summation += coeff[k]

        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense, cosine,  norm])


    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars))

    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [-20000, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])         ##### obj系数

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]

    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1

    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins

    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)
    c_nodes[torch.isnan(c_nodes)] = 1
    return A,  v_nodes, c_nodes

