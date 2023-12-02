import re
import logging
import numpy as np
from math import log, isclose
from itertools import combinations
from decimal import Decimal

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
        self.active_softs = {}
        self.card_encs = {}
        self.factor = 1
        super().__init__(fs_info, solver=solver)

    def _init_soft(self):
        """Create list of soft clauses instead of immediately adding all of them"""
        l, u, I = self._get_index_functions()
        def w(i, j, k, factor):
            d = self.fs_info.get_domain(i)
            i_size = Decimal(d[k])-Decimal(d[j])
            d_size = Decimal(d[-1])-Decimal(d[0])
            return Decimal(log(i_size)) + Decimal(log(factor)) - Decimal(log(d_size))

        all_intervals = [interval for f_intervals in self.interval_sizes.values() for interval in f_intervals]
        while 1/self.factor in all_intervals:
            self.factor += 1

        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            self.active_softs[i] = set()
            for (j, k) in combinations(range(len(d)), 2):
                if j == 0 and k == len(d)-1:
                    self._add_soft(i,j,k)
                    self.card_encs[i] = [[I(i,j,k)]]

    def _expand_softs(self, r_intervals):
        l, u, I = self._get_index_functions()
        for interval in r_intervals:
            i_components = interval.split("_")
            f_id = int(i_components[1])
            l_idx = int(i_components[2])
            u_idx = int(i_components[3])
            reset_cardenc = False
            if u_idx - l_idx > 1:
                if I(f_id,l_idx+1,u_idx) not in self.active_softs[f_id]:
                    self._add_soft(f_id,l_idx+1,u_idx)
                    reset_cardenc = True
                if I(f_id,l_idx,u_idx-1) not in self.active_softs[f_id]:
                    self._add_soft(f_id,l_idx,u_idx-1)
                    reset_cardenc = True
            if reset_cardenc:
                self._reset_cardenc(f_id)
    
    def _add_soft(self, i, j, k):
        def w(i, j, k, factor):
            d = self.fs_info.get_domain(i)
            i_size = Decimal(d[k])-Decimal(d[j])
            d_size = Decimal(d[-1])-Decimal(d[0])
            return Decimal(log(i_size)) + Decimal(log(factor)) - Decimal(log(d_size))

        l, u, I = self._get_index_functions()
        self.wcnf.append([I(i,j,k)], weight=w(i,j,k, self.factor))
        self.active_softs[i].add(I(i,j,k))

    def _reset_cardenc(self, i):
        card_cnf = CardEnc.equals(self.active_softs[i], vpool=self.vpool).clauses
        self.card_encs[i] = card_cnf

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
            for f_id in self.card_encs.keys():
                for c in self.card_encs[f_id]:
                    solver.add_clause(c)
            model = solver.compute()
            if model is None:
                return None
            is_used_interval = lambda x: self.vpool.obj(x) and "I" in self.vpool.obj(x)
            intervals = [self.vpool.obj(x) for x in model if is_used_interval(x)]
            self._expand_softs(intervals)
            bounds = {}
            for I in intervals:
                I = I.split("_")
                f_id = int(I[1])
                l_idx = int(I[2])
                u_idx = int(I[3])
                d = self.fs_info.get_domain(f_id)
                bounds[f_id] = (d[l_idx], d[u_idx])
        return Region(bounds)

    def block_up(self, r):
        super().block_up(r)
        
