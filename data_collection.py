"""
Generate training datas, the datas are of the form: s_t, {A^i_t,f^i_t}
"""
import pickle
import gurobipy as gp
import numpy as np
import argparse
from io import StringIO
from contextlib import redirect_stdout
import os
import pickle
from multiprocessing import Process, Queue
import logging
import random
import sys


def LB_gurobi(ins_path, k0, lb_timelimit, current_sol_dir, bg_dir, sol_dir, threads, log_dir):
    from helper import get_a_new2
    from helper import state_vnode_represent
    from helper import perturb
    #####################################################################################
    """
    read model and initialization
    """
    gp.setParam('LogToConsole', 0)
    A, v_nodes, c_nodes = get_a_new2(ins_path)
    m = gp.read(ins_path)
    m_copy = m.copy()
    name = os.path.basename(ins_path)
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.VarName)

    #####################################################################################
    #####################################################################################
    """
    Initial solution
    """
    m.Params.SolutionLimit = 1
    m.optimize()
    m.resetParams()
    if m.Status == 4 or m.Status == 3:
        return 0
    cur_sol_val = [v.X for v in mvars]
    incumbent = m.ObjVal
    #####################################################################################
    step = 1

    #####################################################################################
    #####################
    """
    iteration of steps of LB LNS
    """

    while True:

        lb = gp.quicksum([v for i, v in enumerate(mvars) if cur_sol_val[i] < 0.5 and v.VType == 'B']) \
             + gp.quicksum([1 - v for i, v in enumerate(mvars) if cur_sol_val[i] > 0.5 and v.VType == 'B'])
        m.addLConstr(lb, gp.GRB.LESS_EQUAL, k0, name='LB')
        m.update()

        log_path = os.path.join(log_dir, os.path.basename(ins_path) + f'_step{step}.log')
        with open(log_path, 'w'):
            pass
        m.Params.LogFile = log_path
        #####################################################################################
        """
        solve LB
        """
        m.Params.Threads = threads
        m.Params.TimeLimit = lb_timelimit
        m.Params.PoolSolutions = 500
        m.Params.PoolSearchMode = 2
        # for idx, v in enumerate(mvars):
        #     v.Start = cur_sol_val[idx]
        m.optimize()
        #####################################################################################
        LB_sol_val = [v.X for v in mvars]
        LBchoice = [int(LB_sol_val[i] != cur_sol_val[i]) if mvars[i].VType == 'B' else 0 for i in range(len(mvars))]
        if m.ObjVal >= incumbent:
            print('LB no improvement')
            break
        else:
            lb_improvement = incumbent - m.ObjVal
        #####saving training data

        BG = [A, state_vnode_represent(v_nodes, cur_sol_val), c_nodes]
        pickle.dump(BG, open(os.path.join(bg_dir, f'{name}_step{step}' + '.bg'), 'wb'))
        pickle.dump(cur_sol_val, open(os.path.join(current_sol_dir, f'{name}_step{step}' + '.cursol'), 'wb'))
        sols = []

        for sn in range(m.SolCount):
            m.Params.SolutionNumber = sn
            sol = [v.Xn for v in mvars]
            choice = [int(sol[i] != cur_sol_val[i]) if mvars[i].VType == 'B' else 0 for i in range(len(mvars))]
            sols.append([choice, incumbent - m.PoolObjVal])

        num = 0
        pert_rate = 0.1
        while num < 500:
            mm = m_copy.copy()
            mm.Params.Threads = 1
            mmvars = mm.getVars()
            mmvars.sort(key=lambda v: v.VarName)
            perturbed_choice = perturb(LBchoice.copy(), pert_rate,
                                       [1 if mmvars[i].Vtype == 'B' else 0 for i in range(len(mmvars))])
            if perturbed_choice in [sol[0] for sol in sols]:
                continue

            for idx, v in enumerate(mmvars):
                if perturbed_choice[idx] == 0 and (v.VType == 'B' or v.VType == 'I'):
                    v.UB = cur_sol_val[idx]
                    v.LB = cur_sol_val[idx]
                v.Start = cur_sol_val[idx]
            mm.update()
            mm.Params.TimeLimit = 60
            mm.optimize()
            num += 1
            pert_rate += 0.01
            sols.append([perturbed_choice, incumbent - mm.ObjVal])
        #################################################################
        # sols = np.array(sols, dtype=np.float32)
        pickle.dump(sols, open(os.path.join(sol_dir, f'{name}_step{step}' + '.sol'), 'wb'))

        cur_sol_val = LB_sol_val
        incumbent -= lb_improvement
        step += 1
        m.remove(m.getConstrByName('LB'))


def collect(ins_dir, qq, settings, bg_dir_c, sol_dir_c, log_dir_c, cur_sol_dir_c):
    radius = settings['k0']
    timelimit = settings['LB timelimit']
    t = settings['threads']
    while True:
        ins_name = qq.get()
        print(ins_name, 'start')
        if not ins_name:
            break
        filepath = os.path.join(ins_dir, ins_name)
        LB_gurobi(ins_path=filepath, k0=radius, lb_timelimit=timelimit,
                  bg_dir=bg_dir_c, sol_dir=sol_dir_c, threads=t, log_dir=log_dir_c, current_sol_dir=cur_sol_dir_c)
        print(ins_name, 'done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nWorkers', type=int, default=60)
    parser.add_argument('--timelimit', type=int, default=2000)
    parser.add_argument('--initial_radius', type=int, default=50)
    parser.add_argument('--instance', type=str, default='MVC')
    parser.add_argument('--Threads', type=int, default=1)
    args = parser.parse_args()
    N_WORKERS = args.nWorkers
    task = args.instance
    instance_task = args.instance

    INS_DIR = f'./instance/{instance_task}/train'

    SOL_DIR = f'./dataset/{task}/r{args.initial_radius}/solution'
    if not os.path.isdir(SOL_DIR):
        os.makedirs(SOL_DIR)
    BG_DIR = f'./dataset/{task}/r{args.initial_radius}/BG'
    if not os.path.isdir(BG_DIR):
        os.makedirs(BG_DIR)
    CUR_SOL_DIR = f'./dataset/{task}/r{args.initial_radius}/current_solution'
    if not os.path.isdir(CUR_SOL_DIR):
        os.makedirs(CUR_SOL_DIR)
    LOG_DIR = f'./dataset/{task}/r{args.initial_radius}/log'
    if not os.path.isdir(LOG_DIR):
        os.makedirs(LOG_DIR)
    SETTINGS = {
        'k0': args.initial_radius,
        'LB timelimit': args.timelimit,
        'threads': args.Threads
    }

    filenames = os.listdir(INS_DIR)
    q = Queue()
    for filename in filenames:
        q.put(filename)
    for i in range(N_WORKERS):
        q.put(None)

    ps = []
    for i in range(N_WORKERS):
        p = Process(target=collect, args=(INS_DIR, q, SETTINGS, BG_DIR, SOL_DIR, LOG_DIR, CUR_SOL_DIR))
        p.start()
        ps.append(p)
    for p in ps:
        p.join()
