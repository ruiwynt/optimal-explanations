from typing import Dict, Tuple, List, Optional
from z3 import *


DECIMAL_PREC = 99

class Region:
    def __init__(self, bounds: Optional[Dict[int, Tuple[float, float]]] = None):
        """
        bounds: {
                    <feature_id>: (<lower>, <upper>),
                    ...
                } 
        
        feature_id should start from 0.
        lower, upper should be floats.
        """
        self.bounds = bounds
        if bounds:
            self.n_features = len(bounds.keys())
    
    def __repr__(self):
        region = [
            "%f <= x%d < %f" % (self.bounds[i][0], i, self.bounds[i][1]) 
            for i in self.bounds.keys()
        ]
        return "\n".join(region)
    
    @classmethod
    def from_model(cls, model, variables):
        """Create region from z3 solver model."""
        region = cls()
        region.bounds = {
            i: (
                float(model[variables[i].lower].as_decimal(DECIMAL_PREC)), 
                float(model[variables[i].upper].as_decimal(DECIMAL_PREC))
            )
            for i in variables.keys()
        }
        region.n_features = len(region.bounds)
        return region


class FeatureVariables:
    def __init__(self, feature_id, vals):
        """Supports real numbers. Implement categorical features sometime later."""
        self.feature_id = feature_id
        self.vals = vals
        self._init_real()
    
    def _init_real(self):
        self.lower = Real('x%d_l' % self.feature_id)
        self.upper = Real('x%d_u' % self.feature_id)

        self.constraints = And(
            self._one_of(self.lower, self.vals),
            self._one_of(self.upper, self.vals),
            self.lower < self.upper,
        )

    def _one_of(self, x, vals):
        return Or([x == i for i in vals])


class FeatureSpaceInfo:
    def __init__(self, thresholds: dict[int: list[float]], domains=None):
        """
        thresholds: {
                    <feature_id>: [<threshold value>, ...],
                    ...
                } 
        """
        self.thresholds = thresholds

        if domains:
            self.domains = domains
        else:
            self.domains = {
                i: (min(thresholds[i])-1, max(thresholds[i])+1)
                for i in thresholds.keys()
            }
    
    def constraint_vals(self, i):
        return [self.domains[i][0]] + self.thresholds[i] + [self.domains[i][1]]
    
    def get_dmin(self, i):
        return self.domains[i][0]

    def get_dmax(self, i):
        return self.domains[i][1]


class ExplanationProgram:
    def __init__(self, thresholds: dict[int: list[float]]):
        self.n_features = len(thresholds)
        self.fs_info = FeatureSpaceInfo(thresholds)
        self.blocked_down = []
        self.blocked_up = []
        self._init_program()

    def __repr__(self):
        s = ""
        s += "n_features: %d\n" % self.n_features
        s += "\n".join([
                "%d: %s" % (i, str(self.fs_info.thresholds[i]))
                for i in self.fs_info.thresholds.keys()
            ]) + "\n"
        s += f"Solver: {self.solver}\n"
        s += f"Blocked Down: {self.blocked_down}\n"
        s += f"Blocked Up: {self.blocked_up}\n"
        return s

    def _init_program(self):
        self.solver = Solver()

        self.vars = {
            i: FeatureVariables(i, self.fs_info.constraint_vals(i))
            for i in self.fs_info.thresholds.keys()
        }

        # Domain Constraints
        for var in self.vars.values():
            self.solver.add(var.constraints)
    
    def get_region(self) -> Region:
        if self.solver.check() == unsat:
            return None
        
        return Region.from_model(self.solver.model(), self.vars)

    def block_down(self, region: Region):
        """Block all regions which are contained within the given region."""
        self.blocked_down.append(region)

        self.solver.add(
            Or([
                Or(self.vars[i].lower < region.bounds[i][0], self.vars[i].upper > region.bounds[i][1])
                for i in self.vars.keys()
            ])
        )

        self.solver.add(self._var_score() > self.get_score(region))

    def block_up(self, region: Region):
        """Block all regions which contain the given region."""
        self.blocked_up.append(region)

        self.solver.add(
            Or([
                Or(self.vars[i].lower > region.bounds[i][0], self.vars[i].upper < region.bounds[i][1])
                for i in self.vars.keys()
            ])
        )
    
    def get_score(self, region: Region):
        """Sum of normalised feature covers."""
        score = 0
        for i in region.bounds.keys():
            # (Length of Region Interval) / (Domain Max - Domain Min)
            score += (region.bounds[i][1] - region.bounds[i][0])/(self.fs_info.get_dmax(i) - self.fs_info.get_dmin(i))
        return score

    def _var_score(self):
        return Sum([
            (v.upper-v.lower)/(self.fs_info.get_dmax(v.feature_id) - self.fs_info.get_dmin(v.feature_id))
            for v in self.vars.values()
        ])


if __name__ == "__main__":
    import random

    random.seed(105523)
    thresholds = {
        i: sorted([random.uniform(0, 1000) for j in range(20)])
        for i in range(3)
    }
    # thresholds = {
    #     0: [1, 2, 3],
    #     1: [1, 2, 3],
    # }
    print('\n'.join([f"{i}: {thresholds[i]}" for i in thresholds.keys()]))

    program = ExplanationProgram(thresholds)

    region = program.get_region()
    while region:
        if random.randint(0, 1) == 1:
            print("Blocking Up")
            program.block_up(region)
        else:
            program.block_down(region)
            print(f"Score: {program.get_score(region)}")
        prev_region = region
        region = program.get_region()
    # print(program)
    print(prev_region)
    print(program.get_score(prev_region))