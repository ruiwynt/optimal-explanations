import time
import bisect
import logging
from math import prod, isclose, log
from copy import deepcopy
from decimal import Decimal, getcontext

from z3 import *

from .entailment import EntailmentChecker
from .regions import Region, FeatureSpaceInfo
from .model import Model
from .generators.z3_generator import SeedGenerator as Z3Generator
from .generators.rc2_generator import SeedGenerator as Rc2Generator
from .generators.greedy_generator import SeedGenerator as GreedyGenerator
from .traverser import LatticeTraverser


class ExplanationProgram:
    def __init__(self, model: Model, limits=None, seed_gen="rand"):
        self.fs_info = FeatureSpaceInfo(model.thresholds, limits=limits)
        self.entailer = EntailmentChecker(model)
        self.seed_gen = seed_gen
        if seed_gen == "rand" or seed_gen == "min":
            self.generator = Z3Generator(self.fs_info, method=seed_gen)
        elif seed_gen == "max":
            # TODO: Rc2Generator doesn't generate largest region as first seed
            self.generator = Rc2Generator(self.fs_info)
        elif seed_gen == "greedy":
            self.generator = GreedyGenerator(self.fs_info)
        else:
            raise ArgumentError(f"{seed_gen} not a valid seed generation method")
        self.traverser = LatticeTraverser(self.entailer, self.fs_info.domains)

        self.total_blocked = 0
        self.n_entailing = 0
        self.n_nonentailing = 0
        self.max_score = -1
        self.max_region = None
        self.seed_score = -1

        self._explain_t = -1
        self._sat_calls = -1
        self._seed_gen_t = -1
        self._traversal_t = -1

    def __repr__(self):
        s = ""
        s += "n_features: %d\n" % self.model.num_feature
        s += "\n".join([
                "%d: %s" % (i, str(self.fs_info.get_domain(i)))
                for i in self.fs_info.keys()
            ]) + "\n"
        s += f"Solver: {self.solver}\n"
        return s
    
    def explain(self, x: list[float]):
        """Find a maximal explanation which contain the instance x."""
        start_t = time.perf_counter()
        self.init_region = self._instance_to_region(x)
        c = self.entailer.predict(x)
        self.traverser.must_contain(self.init_region)
        self.traverser.grow(self.init_region, c) 
        end_t = time.perf_counter()
        self._explain_t = end_t - start_t
        self._sat_calls = self.entailer.oracle_calls
        return self.init_region
    
    def enumerate_explanations(self, x: list[float], block_score=False):
        """Enumerate all maximal explanations for the given class."""
        self.init_region = self._instance_to_region(x)
        c = self.entailer.predict(x)
        self.generator.must_contain(self.init_region)
        self.traverser.must_contain(self.init_region)
        t1 = time.perf_counter()
        r = self.generator.get_seed()
        t2 = time.perf_counter()
        self._seed_gen_t = t2 - t1
        while r:
            # logging.info(f"{self.get_score(r)} | {self.lg_score(r)}")
            score = self.get_score(r)
            self.seed_score = score
            if not self.entailer.entails(r, c):
                # logging.info(f"Non entailing seed generated")
                # logging.info(f"\n{r}")
                t1 = time.perf_counter()
                r = self._instance_to_region(self.entailer.cexample)
                self.traverser.eliminate_vars(r)
                # logging.info(f"Eliminated features\n{r}")
                # logging.info(f"\n{r}")
                t2 = time.perf_counter()
                self._traversal_t = t2 - t1
                self.generator.block_up(r)
                self.n_nonentailing += 1
                if block_score:
                    self._check_entailing_adjacents(r, c)
            else:
                if not (self.seed_gen == "max" or self.seed_gen == "greedy"):
                    t1 = time.perf_counter()
                    self.traverser.grow(r, c)
                    t2 = time.perf_counter()
                    self._traversal_t = t2 - t1
                self._drop_features(r)
                self.generator.block_down(r)
                score = self.get_score(r)
                if block_score:
                    self.generator.block_score(score)
                if score > self.max_score:
                    self.max_score = score
                    self.max_region = r
                self.seed_score = score
                self.n_entailing += 1
                if self.seed_gen == "max" or self.seed_gen == "greedy":
                    logging.info(f"Entailing Seed #{self.n_entailing} | {score:.5f} ")
                    # logging.info(f"\n{r}")
                    if self.n_entailing == 1:
                        self._log_stats()
                        logging.info(f"MAX SCORE: {self.max_score}\n{self.max_region}")
                        return None
            if (self.n_entailing + self.n_nonentailing) % 1 == 0:
                self._log_stats()
            self._sat_calls = self.entailer.oracle_calls
            yield r
            t1 = time.perf_counter()
            r = self.generator.get_seed()
            t2 = time.perf_counter()
            self._seed_gen_t = t2 - t1
        self._log_stats()
        logging.info(f"MAX SCORE: {self.max_score}\n{self.max_region}")

    def get_score(self, r: Region):
        numerator = prod([
            Decimal(r.bounds[i][1])-Decimal(r.bounds[i][0])
            for i in r.bounds.keys()
        ])

        denominator = prod([
            Decimal(self.fs_info.get_dmax(i))-Decimal(self.fs_info.get_dmin(i))
            for i in r.bounds.keys()
        ])

        return numerator/denominator
    
    def reset(self):
        self.generator.reset()
    
    def _log_stats(self):
        s = "Generated "
        s += f"{self.n_entailing + self.n_nonentailing } "
        s += f"({self.n_entailing}E|{self.n_nonentailing}NE) seeds | "
        s += f"{self.entailer.oracle_calls} entailment checks | "
        s += f"max score: {self.max_score:.5f} | "
        s += f"current seed score: {self.seed_score:.5f}"
        logging.info(s)

    def _instance_to_region(self, x: list[float]) -> Region:
        bounds = {}
        for f_id in range(len(x)):
            if f_id not in self.fs_info.keys():
                continue
            d = self.fs_info.get_domain(f_id)
            if x[f_id] in d:
                i = d.index(x[f_id])
                if i == len(d):
                    bounds[f_id] = (d[i-1], d[i])
                else:
                    bounds[f_id] = (d[i], d[i+1])
            else:
                i = bisect.bisect_left(d, x[f_id])
                if i == len(d):
                    bounds[f_id] = (d[i-2], d[i-1])
                elif i == 0:
                    bounds[f_id] = (d[i], d[i+1])
                else:
                    bounds[f_id] = (d[i-1], d[i])
        return Region(bounds)
    
    def _check_entailing_adjacents(self, r: Region, c: str):
        """Checks the entailing regions adjacent to the MinNER r for volume"""
        for f_id in self.fs_info.keys():
            for side in (0, 1):
                d = self.fs_info.get_domain(f_id)
                b = r.bounds[f_id]
                sb = self.traverser.search_bounds[f_id]
                i = d.index(b[0]) if side == 0 else d.index(b[1])

                if side == 0 and i+1 < d.index(b[1]) and i+1 <= sb[0]:
                    r.bounds[f_id] = (d[i+1], b[1])
                elif side == 1 and i-1 > d.index(b[0]) and i-1 >= sb[1]:
                    r.bounds[f_id] = (b[0], d[i-1])
                else:
                    continue
                
                score = self.get_score(r)
                if score > self.max_score:
                    r2 = deepcopy(r)
                    self.traverser.grow(r2, c)
                    self.generator.block_down(r2)
                    self.generator.block_score(score)
                    self.n_entailing += 1
                    self.max_score = score
                    self.max_region = r2
                r.bounds[f_id] = b
    
    def _drop_features(self, r: Region):
        to_remove = set()
        for f_id, b in r.bounds.items():
            d = self.fs_info.get_domain(f_id)
            if isclose(b[0], d[0]) and isclose(b[1], d[-1]):
                to_remove.add(f_id)
        for f_id in to_remove:
            del r.bounds[f_id]
            