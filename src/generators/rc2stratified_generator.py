import re
import logging
import numpy as np
import heapq
from math import log, isclose
from itertools import combinations
from decimal import Decimal

from pysat.formula import WCNFPlus, IDPool
from pysat.card import CardEnc
from pysat.examples.rc2 import RC2

from src.regions import Region
from src.utils.sat_shortcuts import *
from src.generators.rc2_generator import SeedGenerator as RC2Generator


class SeedGenerator(RC2Generator):
    """
    Generate unblocked seed with maximum volume.
    """
    def __init__(self, fs_info, solver="g4"):
        self.soft_pq = []
        self.active_softs = {}
        self.card_encs = None
        self.level_size = int(fs_info.n_pairs()/len(fs_info.keys()))
        self.current_level = 0
        super().__init__(fs_info, solver=solver)

    def _init_soft(self):
        """Create list of soft clauses instead of immediately adding all of them"""
        l, u, I = self._get_index_functions()
        def w(i, j, k, factor):
            d = self.fs_info.get_domain(i)
            i_size = Decimal(d[k])-Decimal(d[j])
            d_size = Decimal(d[-1])-Decimal(d[0])
            return Decimal(log(i_size)) + Decimal(log(factor)) - Decimal(log(d_size))

        factor = 1
        all_intervals = [interval for f_intervals in self.interval_sizes.values() for interval in f_intervals]
        while 1/factor in all_intervals:
            factor += 1
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            for (j, k) in combinations(range(len(d)), 2):
                self.soft_pq.append((-w(i,j,k, factor), i, j, k, I(i,j,k)))
        heapq.heapify(self.soft_pq)
        self._activate_intervals()

    def _activate_intervals(self):
        if len(self.soft_pq) == 0:
            return False
        i = 0
        n_pops = min(self.level_size, len(self.soft_pq))
        while i < n_pops:
            interval = heapq.heappop(self.soft_pq)
            w, f_id, var = -interval[0], interval[1], interval[4]
            self.wcnf.append([var], weight=w)

            if not f_id in self.active_softs.keys():
                self.active_softs[f_id] = []
            self.active_softs[f_id].append(var)
            i += 1
        self.card_encs = []
        for f_id in self.active_softs.keys():
            ivars = self.active_softs[f_id] 
            self.card_encs += CardEnc.equals(ivars, vpool=self.vpool).clauses
        self.current_level += 1
        n_added = self.current_level*self.level_size
        N = self.fs_info.n_pairs()
        logging.info(f"Activated new clauses. Progress: {n_added}/{N} ({100*n_added/N:.2f}%)")
        return True

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
            for c in self.card_encs:
                solver.add_clause(c)
            model = solver.compute()
            while model is None:
                solver.delete()
                if not self._activate_intervals():
                    return None
                solver.init(self.wcnf, incr=True)
                for c in self.card_encs:
                    solver.add_clause(c)
                model = solver.compute()
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

