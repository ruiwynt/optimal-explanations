import re
import logging
import numpy as np
from math import log, isclose
from itertools import combinations
from decimal import Decimal

from pysat.formula import WCNFPlus, IDPool
from pysat.card import CardEnc
from pysat.examples.rc2 import RC2

from src.regions import Region
from src.utils.sat_shortcuts import *


class SeedGenerator:
    """
    Generate unblocked seed with maximum volume.
    """
    def __init__(self, fs_info, solver="g4"):
        self.fs_info = fs_info
        self.vpool = IDPool(start_from=1)
        self.wcnf = WCNFPlus()
        self.solver = solver
        self.interval_sizes = {}
        # self.constraints = []

        self._init_hard_bounds()
        self._init_hard_intervals()
        self._init_soft()
        self.rc2 = RC2(
            self.wcnf, 
            solver=solver, 
            adapt=True,
            exhaust=True,
            minz=True,
            trim=True,
        )

        self.n_vars = len(self.vpool.obj2id.keys())
        self.n_clauses = len(self.wcnf.hard) + len(self.wcnf.soft)
    
    def _init_hard_bounds(self):
        l, u, I = self._get_index_functions()
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            for j in range(len(d)):
                l(i,j)  # l_ij <-> d[j] is lower bound
                u(i,j)  # u_ij <-> d[j] is upper bound
            self.wcnf.append([-l(i,len(d)-1)])  # Lower bound can't be highest threshold
            # self.constraints.append(f"~{l(i,len(d)-1)}")
            self.wcnf.append([-u(i,0)])  # Upper bound can't be lowest threshold
            # self.constraints.append(f"~{u(i,0)}")
            for j in range(1, len(d)-1):
                l_lt_u = Implies(
                    l(i,j),
                    Not(Or([u(i,k) for k in range(j+1)]))
                )  # l_ij -> ~(u_i0 v u_i1 v ... v u_ij)
                # self.constraints.append(l_lt_u)
                self.wcnf.extend(l_lt_u.to_cnf())

                u_gt_t = Implies(
                    u(i,j),
                    Not(Or([l(i,k) for k in range(j, len(d))]))
                )  # u_ij -> ~(l_ij v l_i(j+1) v ... v l_im)
                # self.constraints.append(u_gt_t)
                self.wcnf.extend(u_gt_t.to_cnf())
        
    def _init_hard_intervals(self):
        l, u, I = self._get_index_functions()
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            sizes = []
            for (j, k) in combinations(range(len(d)), 2):
                I(i,j,k)  # I_ijk <-> interval is (d[j], d[k])
                constraint = Iff(And([l(i,j),u(i,k)]), I(i,j,k))
                # self.constraints.append(constraint)
                self.wcnf.extend(constraint.to_cnf())  # (l_ij ^ u_ik) <-> I_ijk
                sizes.append(d[k]-d[j])
            I_vars = [I(i,j,k) for (j, k) in combinations(range(len(d)), 2)]
            # self.constraints.append(f"sum({I_vars}) = 1")
            card = CardEnc.equals(I_vars, vpool=self.vpool).clauses
            self.wcnf.extend(card)  # Exactly one I_ijk 
            self.interval_sizes[i] = sorted(sizes, reverse=True)

    def _init_soft(self):
        l, u, I = self._get_index_functions()
        def w(i, j, k, factor):
            d = self.fs_info.get_domain(i)
            i_size = Decimal(d[k])-Decimal(d[j])
            return Decimal(log(i_size)) + Decimal(log(factor))

        factor = 1
        all_intervals = [interval for f_intervals in self.interval_sizes.values() for interval in f_intervals]
        while 1/factor in all_intervals:
            factor += 1
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            for (j, k) in combinations(range(len(d)), 2):
                self.wcnf.append([I(i,j,k)], weight=w(i,j,k, factor))

    def get_seed(self) -> Region:
        with RC2(
            self.wcnf, 
            solver=self.solver, 
            adapt=True,
            exhaust=True,
            incr=True,
            minz=True,
            trim=True,
        ) as solver:
            model = solver.compute()
            if model is None:
                logging.info("UNSAT")
                solver.get_core()
                logging.info(f"UNSAT Core: {solver.core}")
                return None

            is_used_interval = lambda x: self.vpool.obj(x) and "I" in self.vpool.obj(x)
            intervals = [self.vpool.obj(x) for x in model if is_used_interval(x)]
            bounds = {}
            for I in intervals:
                I = I.split("_")
                f_id = int(I[1])
                l_idx = int(I[2])
                u_idx = int(I[3])
                d = self.fs_info.get_domain(f_id)
                bounds[f_id] = (d[l_idx], d[u_idx])
            return Region(bounds)
        # model = self.rc2.compute()
        # if model is None:
        #     logging.info("UNSAT")
        #     self.rc2.get_core()
        #     logging.info(f"UNSAT Core: {self.rc2.core}")
        #     return None
        # is_used_interval = lambda x: self.vpool.obj(x) and "I" in self.vpool.obj(x)
        # intervals = [self.vpool.obj(x) for x in model if is_used_interval(x)]
        # bounds = {}
        # for I in intervals:
        #     I = I.split("_")
        #     f_id = int(I[1])
        #     l_idx = int(I[2])
        #     u_idx = int(I[3])
        #     d = self.fs_info.get_domain(f_id)
        #     bounds[f_id] = (d[l_idx], d[u_idx])
        # return Region(bounds)

    def must_contain(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_conjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            u_idx = d_idx[i][1]
            to_conjunct.append(Or([l(i,j) for j in range(l_idx+1)]))
            to_conjunct.append(Or([u(i,k) for k in range(u_idx, len(d))]))
        # self.constraints.append(And(to_conjunct))
        self.wcnf.extend(And(to_conjunct).to_cnf()) 
        self._extend_rc2(And(to_conjunct).to_cnf())

    def block_up(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            if l_idx < len(d)-1:
                to_disjunct.append(Or([l(i,j) for j in range(l_idx+1, len(d))]))
            u_idx = d_idx[i][1]
            if u_idx > 0:
                to_disjunct.append(Or([u(i,k) for k in range(u_idx)]))
        # self.constraints.append(Or(to_disjunct))
        self.wcnf.extend(Or(to_disjunct).to_cnf()) 
        self._extend_rc2(Or(to_disjunct).to_cnf())

    def block_down(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            if l_idx > 0:
                to_disjunct.append(Or([l(i,j) for j in range(l_idx)]))
            u_idx = d_idx[i][1]
            if u_idx < len(d)-1:
                to_disjunct.append(Or([u(i,k) for k in range(u_idx+1, len(d))]))
        # self.constraints.append(Or(to_disjunct))
        self.wcnf.extend(Or(to_disjunct).to_cnf()) 
        self._extend_rc2(Or(to_disjunct).to_cnf())
    
    def _print_constraints(self):
        for c in self.constraints:
            self._print_enc(c)
    
    def _print_enc(self, s):
        s = str(s)
        d = {str(k): o for (k, o) in self.vpool.id2obj.items()}
        keys = (re.escape(k) for k in d.keys())
        pattern = re.compile(r'\b(' + '|'.join(keys) + r')\b')
        print(pattern.sub(lambda x: d[x.group()], s))
    
    def _extend_rc2(self, cnf):
        for clause in cnf:
            self.rc2.add_clause(clause)

    def _region_to_didx(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = {}
        for i, b in r.bounds.items(): 
            d = self.fs_info.get_domain(i)
            d_idx[i] = (d.index(b[0]), d.index(b[1]))
        return d_idx
    
    def _get_index_functions(self):
        l = lambda i, j: self.vpool.id(f"l_{i}_{j}")
        u = lambda i, j: self.vpool.id(f"u_{i}_{j}")
        I = lambda i, j, k: self.vpool.id(f"I_{i}_{j}_{k}")
        return l, u, I
