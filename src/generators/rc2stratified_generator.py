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
        self.stratified_softs = {}
        self.next_level = 0
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
                level = len(d) - (k - j) - 1
                if not level in self.stratified_softs.keys():
                    self.stratified_softs[level] = {"softs": [], "ivars": {f_id: [] for f_id in self.fs_info.keys()}, "card": []}
                self.stratified_softs[level]["softs"].append(([I(i,j,k)], w(i,j,k,factor)))
                self.stratified_softs[level]["ivars"][i].append(I(i,j,k))
        for level in self.stratified_softs.keys():
            for i in self.fs_info.keys():
                ivars = self.stratified_softs[level]["ivars"][i]
                if len(ivars) > 0:
                    self.stratified_softs[level]["card"] += CardEnc.equals(ivars, vpool=self.vpool).clauses
        self._start_next_level()

    def _start_next_level(self, solver=None):
        if not self.next_level in self.stratified_softs.keys():
            return False
        for c, w in self.stratified_softs[self.next_level]["softs"]:
            self.wcnf.append(c, weight=w)
        self.next_level += 1
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
            for c in self.stratified_softs[self.next_level-1]["card"]:
                solver.add_clause(c)
            model = solver.compute()
            while model is None:
                solver.delete()
                if not self._start_next_level():
                    return None
                solver.init(self.wcnf, incr=True)
                for c in self.stratified_softs[self.next_level-1]["card"]:
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

