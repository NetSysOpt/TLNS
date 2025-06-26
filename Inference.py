import os
import pickle
import gurobipy as gp
import pyscipopt as scp
import time
from loguru import logger
import numpy as np
import argparse
import random


def double_LNS_scip(ins_name, timelimit, sub_timelimit, initial_radius, initial_inner_radius,
                    inner_adaptive_radius, inner_times, model_path=None, adaptive_radius=1.02,
                    log_path=None, random_seed=0, destroy_method='ml', tolerance=1e-6):
    import torch
    import pyscipopt as scp
    from helper import get_a_new2, state_vnode_represent, choose, fix_seed
    from loguru import logger
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDE_OUTPUT = True
    if log_path is not None:
        handler = logger.add(os.path.join(log_path), level='DEBUG')
    fix_seed(random_seed)
    timing = []
    incumbents = []
    m_stuck_times = 0

    ####  READING ####
    m = scp.Model()
    m.readProblem(ins_name)
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)

    m.setParam('parallel/maxnthreads', 1)
    m.setParam('randomization/randomseedshift', random_seed)
    m.setParam('randomization/lpseed', random_seed)
    m.setParam('randomization/permutationseed', random_seed)
    m.setParam('concurrent/changeseeds', False)
    m.hideOutput(HIDE_OUTPUT)

    if destroy_method == 'ml':
        logger.info('parsing for GNN representation')
        from GCN import SGT as policy
        gnn = policy()
        gnn.load_state_dict(torch.load(model_path))
        gnn.to(DEVICE)
        gnn.eval()
        A, v_nodes, c_nodes = get_a_new2(ins_name)
        logger.info('parsing done')

    start = time.monotonic()
    event = MyEvent()
    ### INITIAL SOLUTION ###
    logger.info(f'GRB for initial solution')
    gp.setParam('LogToConsole', 0)
    gm = gp.read(ins_name)
    gm.Params.SolutionLimit = 1
    gm.Params.Seed = random_seed
    gm.Params.Threads = 1
    gm.optimize()
    vars = gm.getVars()
    vars.sort(key=lambda v: v.VarName)
    cur_sol_val = [v.X for v in vars]
    incumbent = gm.ObjVal
    logger.info(f'Initial solution found, obj={incumbent}/ total time:{time.monotonic() - start}')
    timing.append(time.monotonic() - start)
    incumbents.append(incumbent)

    step = 0
    k = initial_radius
    m_incumbent = incumbent
    dty, fname = os.path.split(ins_name)
    presolved_path = os.path.join(os.path.split(dty)[0], 'presolved', os.path.basename(fname))
    if not os.path.isdir(os.path.split(presolved_path)[0]):
        os.makedirs(os.path.split(presolved_path)[0])

    while True:
        if destroy_method == 'ml':
            edge_features = A._values().unsqueeze(1)
            with torch.no_grad():
                gnn.eval()
                score = gnn(c_nodes.to(DEVICE),
                            A._indices().to(DEVICE),
                            edge_features.to(DEVICE),
                            state_vnode_represent(v_nodes, cur_sol_val).to(DEVICE))
                score = score.sigmoid()
                score = score.to('cpu').numpy()
                m_fixation, m_destroy_choice = choose(score, int(k))
                torch.cuda.empty_cache()
        elif destroy_method == 'random':
            m_destroy_choice = random.sample(range(len(mvars)), int(k))
            m_fixation = list(set([i for i in range(len(mvars))]) - set(m_destroy_choice))

        logger.debug(f'start solving subMIP of size {len(m_destroy_choice)} / total time:{time.monotonic() - start}')

        ### m fix    ###
        m_lower_bound = {}
        m_upper_bound = {}
        for idx in m_fixation:
            v = mvars[idx]
            m_upper_bound[idx] = v.getUbOriginal()
            m_lower_bound[idx] = v.getLbOriginal()
            m.chgVarLb(v, cur_sol_val[idx])
            m.chgVarUb(v, cur_sol_val[idx])
        cur_sol = m.createSol()
        for idx, v in enumerate(mvars):
            m.setSolVal(cur_sol, v, cur_sol_val[idx])
        m.addSol(cur_sol, free=False)

        m.setParam('constraints/linear/upgrade/setppc', False)
        m.setParam('constraints/linear/upgrade/logicor', False)
        m.setParam('constraints/linear/upgrade/indicator', False)
        m.setParam('constraints/linear/upgrade/knapsack', False)
        m.setParam('constraints/linear/upgrade/xor', False)
        m.setParam('constraints/linear/upgrade/varbound', False)

        # m.setParam('presolving/donotaggr', True)
        # m.setParam('presolving/donotmultaggr', True)

        sp = time.monotonic()
        m.presolve()

        m.writeProblem(presolved_path, trans=True)
        p = scp.Model()
        p.readProblem(presolved_path)
        p.hideOutput(HIDE_OUTPUT)
        pvars = p.getVars()
        pvars.sort(key=lambda v: v.name)
        m_getTransedVarsByName = {v.name: v for v in m.getVars(True)}
        p_getVarsByName = {v.name: v for v in pvars}
        m_getVarByTransedName = {m.getTransformedVar(v).name: v for v in m.getVars()}
        if m.getStage() == 10:
            logger.debug('presolve finish solve')
            cur_sol = m.getBestSol()
            timing.append(time.monotonic() - start)
            incumbent = m.getSolObjVal(cur_sol)
            m_incumbent = incumbent
            incumbents.append(incumbent)
            cur_sol_val = [cur_sol[v] for v in mvars]
            k *= 1.1
            logger.debug('to next turn')
        else:
            logger.debug(f'TLNS start, total time={time.monotonic() - start}')
            kk = min(initial_inner_radius, len(pvars))
            if destroy_method == 'ml':
                A_p, v_nodes_p, c_nodes_p = get_a_new2(presolved_path)
                c_nodes_p[torch.isnan(c_nodes_p)] = 1

            ########################################  sub MIP & sub LNS -------------------------------------
            penalty = 0
            ### create an initial solution of p
            ### cur_sol is a feasible solution of m, but may not correspond to a feasible solution in p
            ### dont know how to convert cur_sol to cur_sol_p
            cur_sol_p = p.createSol()
            for v in pvars:
                p.setSolVal(cur_sol_p, v, cur_sol[m_getTransedVarsByName[v.name]])
            p.addSol(cur_sol_p, free=False)
            cur_sol_val_p = [cur_sol_p[v] for v in pvars]

            #################################
            ########    LNS for p    ########
            #################################
            while True:
                if destroy_method == 'ml':
                    edge_features_p = A_p._values().unsqueeze(1)
                    with torch.no_grad():
                        score_p = gnn(c_nodes_p.to(DEVICE),
                                      A_p._indices().to(DEVICE),
                                      edge_features_p.to(DEVICE),
                                      state_vnode_represent(v_nodes_p, cur_sol_val_p).to(DEVICE))
                        score_p = score_p.sigmoid()
                        score_p = score_p.to('cpu').numpy()
                        fixation, destroy_choice = choose(score_p, int(kk))
                        torch.cuda.empty_cache()
                elif destroy_method == 'random':
                    destroy_choice = random.sample(range(len(pvars)), int(kk))
                    fixation = list(set([i for i in range(len(pvars))]) - set(destroy_choice))
                ###########################################################
                #####              FIXING  &  MIP start           #########
                ###########################################################
                pub = {}
                plb = {}
                for idx in fixation:
                    v = pvars[idx]
                    pub[idx] = v.getUbOriginal()
                    plb[idx] = v.getLbOriginal()
                    p.chgVarUb(v, cur_sol_val_p[idx])
                    p.chgVarLb(v, cur_sol_val_p[idx])

                ###########################################################
                #####                      SOLVING                #########
                ###########################################################
                p.setParam('limits/time', sub_timelimit)
                p.setParam('parallel/maxnthreads', 1)
                p.setParam('randomization/randomseedshift', random_seed)
                p.setParam('randomization/lpseed', random_seed)
                p.setParam('randomization/permutationseed', random_seed)
                p.optimize()
                cur_sol_p = p.getBestSol()
                cur_sol_val_p = [cur_sol_p[v] for v in pvars]
                cur_sol_val_p_dict = {v.name: cur_sol_p[v] for v in pvars}
                cur_obj = p.getSolObjVal(cur_sol_p)
                logger.debug(f'TLNS-intermediate sol:{cur_obj} / total time={time.monotonic() - start}')

                if cur_obj > incumbent - tolerance:
                    penalty += 1
                    kk = min(inner_adaptive_radius * kk, len(pvars))
                else:
                    incumbent = cur_obj
                timing.append(time.monotonic() - start)
                incumbents.append(incumbent)

                if penalty >= inner_times:
                    if cur_obj <= m_incumbent + tolerance:
                        break
                p.freeTransform()
                ######################################
                #####        unfix                ####
                ######################################
                for idx in fixation:
                    v = pvars[idx]
                    p.chgVarUb(v, pub[idx])
                    p.chgVarLb(v, plb[idx])

            #################################
            ####   map back to m     ########
            #################################
            for v in pvars:
                m.setSolVal(cur_sol, m_getVarByTransedName[v.name], cur_sol_p[v])
                m.chgVarUb(m_getVarByTransedName[v.name], cur_sol_p[v])
                m.chgVarLb(m_getVarByTransedName[v.name], cur_sol_p[v])
            for v in mvars:
                if v.getLbGlobal() == v.getLbGlobal():
                    m.setSolVal(cur_sol, v, v.getLbGlobal())

            m.addSol(cur_sol, free=False)
            cur_sol_val = [cur_sol[v] for v in mvars]

            if abs(m_incumbent - incumbent) < tolerance:
                k = min(k * adaptive_radius, len(mvars))
                inner_adaptive_radius = min(1.05 * inner_adaptive_radius, 1.8)
                m_stuck_times += 1

            m_incumbent = incumbent
            if not m.checkSol(cur_sol):
                logger.warning('cur_sol is infeasible')
                quit()
            if abs(p.getSolObjVal(cur_sol_p) - m.getSolObjVal(cur_sol)) > tolerance:
                logger.warning('p!=m')
                quit()
        logger.debug(
            f'subMIP solved, total time={time.monotonic() - start} incumbent={incumbents[-1]}')

        step += 1

        if time.monotonic() - start > timelimit:
            if log_path is not None:
                logger.remove(handler)
            return timing, incumbents, cur_sol_val

        m.freeTransform()
        for v in mvars:
            m.chgVarUb(v, 1)
            m.chgVarLb(v, 0)


def single_LNS_scip(ins_name, timelimit, sub_timelimit, initial_radius, destroy_method, adaptive_radius, beta=1,
                    model_path=None, log_path=None, random_seed=0, tolerance=1e-6):
    import random
    import torch
    import pyscipopt as scp
    from helper import get_a_new2, state_vnode_represent, choose, fix_seed
    from loguru import logger
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDE_OUTPUT = True
    if log_path is not None:
        handler = logger.add(os.path.join(log_path), level='INFO')
    logger.info(f'{ins_name} start')
    logger.info(f'initial radius{initial_radius}subtimelimit{sub_timelimit}model{model_path}')
    fix_seed(random_seed)
    timing = []
    incumbents = []

    ### Reading
    m = scp.Model()
    m.readProblem(ins_name)
    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)
    m.setParam('parallel/maxnthreads', 1)
    if destroy_method == 'ml':
        from GCN import SGT as policy
        gnn = policy()
        gnn.load_state_dict(torch.load(model_path))
        gnn.to(DEVICE)
        gnn.eval()
        A, v_nodes, c_nodes = get_a_new2(ins_name)
        c_nodes[torch.isnan(c_nodes)] = 1

    m.hideOutput(HIDE_OUTPUT)

    start = time.monotonic()
    event = MyEvent()
    m.includeEventhdlr(
        event,
        "",
        ""
    )

    ###initial solution
    logger.info(f'GRB for initial solution')
    gp.setParam('LogToConsole', 0)
    gm = gp.read(ins_name)
    gm.Params.SolutionLimit = 1
    gm.Params.Seed = random_seed
    gm.Params.Threads = 1
    gm.optimize()
    vars = gm.getVars()
    vars.sort(key=lambda v: v.VarName)
    cur_sol_val = [v.X for v in vars]
    incumbent = gm.ObjVal
    logger.info(f'Initial solution found, obj={incumbent}')
    timing.append(time.monotonic() - start)
    incumbents.append(incumbent)

    _step = 0
    k = initial_radius

    while True:
        if destroy_method == 'random':
            destroy_choice = random.sample(range(len(mvars)), int(k))
            fixation = list(set([i for i in range(len(mvars))]) - set(destroy_choice))

        elif destroy_method == 'ml':
            with torch.no_grad():
                gnn.eval()
                edge_features = A._values().unsqueeze(1)
                score = gnn(c_nodes.to(DEVICE),
                            A._indices().to(DEVICE),
                            edge_features.to(DEVICE),
                            state_vnode_represent(v_nodes, cur_sol_val).to(DEVICE))
                score = score.sigmoid()
                score = score.to('cpu').numpy()
                fixation, destroy_choice = choose(score, int(k))
                torch.cuda.empty_cache()

        #########   m fix
        m_lower_bound = {}
        m_upper_bound = {}
        for idx in fixation:
            v = mvars[idx]
            m_upper_bound[idx] = v.getUbOriginal()
            m_lower_bound[idx] = v.getLbOriginal()
            m.chgVarLb(v, cur_sol_val[idx])
            m.chgVarUb(v, cur_sol_val[idx])

        ##### m.start (hint)
        cur_sol = m.createSol()
        for idx, v in enumerate(mvars):
            m.setSolVal(cur_sol, v, cur_sol_val[idx])
        m.addSol(cur_sol, free=False)

        ### solving sub-MIP
        sub_mip_start = time.monotonic()
        m.setParam('limits/time', sub_timelimit)
        m.setParam('parallel/maxnthreads', 1)
        m.setParam('randomization/randomseedshift', random_seed)
        m.setParam('randomization/lpseed', random_seed)
        m.setParam('randomization/permutationseed', random_seed)
        logger.info(f'subMIP start, radius={len(destroy_choice)}')
        m.optimize()
        timing += [abs_t - start for abs_t in event.abs_timing]
        incumbents += event.incumbents

        cur_sol = m.getBestSol()
        cur_obj = m.getSolObjVal(cur_sol)
        logger.info(f'subMIP solving time: {time.monotonic() - sub_mip_start} / total time: {time.monotonic() - start}')

        if cur_obj > incumbent + tolerance:
            logger.warning(f'current solution worse than incumbent')
            quit()
        elif cur_obj < incumbent - tolerance:
            incumbent = cur_obj
        else:
            k = min(k * adaptive_radius, beta * len(mvars))
        logger.info(f'current obj val: {cur_obj}')
        cur_sol_val = [cur_sol[v] for v in mvars]

        if time.monotonic() - start > timelimit:
            if log_path is not None:
                logger.remove(handler)
            return timing, incumbents

        m.freeTransform()
        _step += 1
        for idx in fixation:
            v = mvars[idx]
            m.chgVarUb(v, m_upper_bound[idx])
            m.chgVarLb(v, m_lower_bound[idx])


def gurobi(ins_name, timelimit, log_path=None, seed=0):
    import gurobipy as gp
    m = gp.read(ins_name)
    if log_path is not None:
        m.Params.LogFile = log_path

    m.Params.Threads = 1
    m.Params.TimeLimit = timelimit
    m.Params.Seed = seed
    m.Params.DisplayInterval = 50
    incumbents = []
    timing = []

    def callback(model, where):
        if where == gp.GRB.Callback.MIPSOL:
            timing.append(model.cbGet(gp.GRB.Callback.RUNTIME))
            incumbents.append(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))

    m.optimize(callback)
    return timing, incumbents


class MyEvent(scp.Eventhdlr):
    def eventinit(self):
        self.timing = []
        self.abs_timing = []
        self.incumbents = []
        self.start_time = time.monotonic()
        self.model.catchEvent(scp.SCIP_EVENTTYPE.BESTSOLFOUND, self)

    def eventexec(self, event):
        self.sol_found_time = time.monotonic()
        sol = self.model.getBestSol()
        obj = self.model.getSolObjVal(sol)
        self.timing.append(self.sol_found_time - self.start_time)
        self.abs_timing.append(self.sol_found_time)
        self.incumbents.append(obj)


def scip(ins_name, time_limit, log_path=None, seed=0):
    model = scp.Model()
    model.readProblem(ins_name)
    # model.hideOutput(True)
    model.setParam("limits/time", time_limit)
    model.setParam('parallel/maxnthreads', 1)
    model.setParam('randomization/randomseedshift', seed)
    model.setParam('randomization/lpseed', seed)
    model.setParam('randomization/permutationseed', seed)
    model.setHeuristics(scp.SCIP_PARAMSETTING.AGGRESSIVE)  ##### use heuristics aggressively
    model.setLogfile(log_path)
    event = MyEvent()
    model.includeEventhdlr(
        event,
        "",
        ""
    )
    model.optimize()
    return event.timing, event.incumbents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--timelimit', type=int, default=1000)
    parser.add_argument('--instance', type=str, default='SC')
    parser.add_argument('--solver', type=str, default='TLNS', choices=['TLNS', 'LNS', 'gurobi', 'scip'])
    parser.add_argument('--destroy', type=str, default='ml', choices=['ml', 'random'])
    parser.add_argument('--k', type=int, default=None)
    parser.add_argument('--kk', type=int, default=None)
    parser.add_argument('--inner', type=int, default=4)
    parser.add_argument('--seed', type=int, default=307)
    args = parser.parse_args()

    if args.solver in ['TLNS', 'LNS']:
        method = f'{args.destroy}-{args.solver}'
    else:
        method = args.solver
    SETTINGS = {
        'timelimit': args.timelimit,
        'destroy_method': args.destroy,
        'model_path': f'./model/{args.instance}/{args.instance}.pth',
        'result_path': f'./result/{method}/{args.instance}',
        'initial_radius': args.k,
        'initial_inner_radius': args.kk,
        'inner_times': args.inner,
        'seed': args.seed
    }

    if not os.path.exists(SETTINGS['result_path']):
        os.makedirs(SETTINGS['result_path'])

    for i in range(10):
        if os.path.exists(f"{SETTINGS['result_path']}/instance_{i + 1}.txt"):
            continue
        ins_path = f'./instance/{args.instance}/test/instance_{i + 1}.lp'
        if args.solver == 'gurobi':
            result = gurobi(ins_name=ins_path, timelimit=SETTINGS['timelimit'],
                            seed=SETTINGS['seed'])

        elif args.solver == 'scip':
            result = scip(ins_name=ins_path, time_limit=SETTINGS['timelimit'], seed=SETTINGS['seed'])

        elif args.solver == 'TLNS':
            result = double_LNS_scip(ins_name=ins_path, timelimit=1000,
                                     sub_timelimit=5,
                                     initial_radius=SETTINGS['initial_radius'], destroy_method=SETTINGS['destroy_method'],
                                     initial_inner_radius=SETTINGS['initial_inner_radius'],
                                     inner_adaptive_radius=1.15, inner_times=SETTINGS['inner_times'],
                                     adaptive_radius=1.05, model_path=SETTINGS['model_path'],
                                     random_seed=SETTINGS['seed'])
        elif args.solver == 'LNS':
            result = single_LNS_scip(ins_name=ins_path, timelimit=1000, sub_timelimit=50, initial_radius=SETTINGS['initial_radius'],
                                     adaptive_radius=1.05, destroy_method=SETTINGS['destroy_method'], model_path=SETTINGS['model_path'])

        with open(f"{SETTINGS['result_path']}/instance_{i + 1}.txt", "w") as file:
            file.writelines(str(result))
