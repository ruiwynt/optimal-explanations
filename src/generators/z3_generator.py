import logging
from math import prod

from z3 import *

from ..regions import Region, FeatureSpaceInfo, LimitVariables


class SeedGenerator:
    def __init__(
            self, 
            fs_info: FeatureSpaceInfo, 
            score: str="volume", 
            method: str="rand",
            solver: str="z3"
        ):
        self.score = score
        self.method = method
        self.fs_info = fs_info

        if method == "min":
            self.solver = Optimize()
        else:
            self.solver = Solver()

        self.vars = {
            i: LimitVariables(i, fs_info.get_domain(i))
            for i in fs_info.keys()
        }

        # Domain constraints
        for var in self.vars.values():
            self.solver.add(var.constraints)

    def get_seed(self) -> Region:
        if self.method == "min":
            logging.debug(f"Adding minimisation objective to solver...")
            self.solver.minimize(self._var_score())
            logging.debug(f"Objective added...")
        if self.solver.check() == unsat:
            logging.info(f"All regions explored.")
            return None
        self.region = Region.from_z3model(self.solver.model(), self.vars)
        return self.region

    def must_contain(self, r: Region):
        self.solver.add(
            And([
                And(
                    self.vars[i].lower <= r.bounds[i][0], 
                    self.vars[i].upper >= r.bounds[i][1]
                )
                for i in self.vars.keys()
            ])
        )

    def block_up(self, r: Region):
        """Block all regions which contain the given region."""
        self.solver.add(
            Or([
                Or(
                    self.vars[i].lower > r.bounds[i][0], 
                    self.vars[i].upper < r.bounds[i][1]
                )
                for i in self.vars.keys()
            ])
        )

    def block_down(self, r: Region):
        """Block all regions which are contained within the given region."""
        self.solver.add(
            Or([
                Or(
                    self.vars[i].lower < r.bounds[i][0], 
                    self.vars[i].upper > r.bounds[i][1]
                )
                for i in self.vars.keys()
            ])
        )

    def block_score(self, score: float):
        self.solver.add(self._var_score() > score)
    
    def reset(self):
        if self.method == "min":
            self.solver = Optimize()
        else:
            self.solver = Solver()
    
    def _var_score(self):
        interval_sizes = [
            (v.upper-v.lower)
            for v in self.vars.values()
        ]

        domain_sizes = [
            self.fs_info.get_dmax(v.feature_id)-self.fs_info.get_dmin(v.feature_id)
            for v in self.vars.values()
        ]

        return Product([
            (interval_sizes[i] / domain_sizes[i])
            for i in range(len(interval_sizes))
        ])

    # def get_add_score(self, r: Region):
    #     """Sum of normalised feature covers."""
    #     score = 0
    #     for i in r.bounds.keys():
    #         # (Length of r Interval) / (Domain Max - Domain Min)
    #         score += (r.bounds[i][1] - r.bounds[i][0])/(self.fs_info.get_dmax(i) - self.fs_info.get_dmin(i))
    #     return score

    # def _var_add_score(self):
    #     return Sum([
    #         (v.upper-v.lower)/(self.fs_info.get_dmax(v.feature_id) - self.fs_info.get_dmin(v.feature_id))
    #         for v in self.vars.values()
    #     ])