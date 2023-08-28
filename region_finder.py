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


class ExplanationProgram:
    def __init__(self, splits: Dict[int, List[float]]):
        """
        splits: {
                    <feature_id>: [<threshold value>, ...],
                    ...
                } 
        """
        self.n_features = len(splits)
        self.splits = splits  # Should implement sentinel values -inf and inf
        self._init_program()

    def __repr__(self):
        s = ""
        s += "n_features: %d\n" % self.n_features
        s += "\n".join([
                "%d: %s" % (i, str(self.splits[i]))
                for i in self.splits.keys()
            ]) + "\n"
        s += str(self.solver)
        return s

    def _init_program(self):
        self.solver = Solver()

        self.vars = {
            i: FeatureVariables(i, self.splits[i])
            for i in self.splits.keys()
        }

        # Variable Constraints
        for var in self.vars.values():
            self.solver.add(var.constraints)
    
    def solve(self) -> Region:
        if self.solver.check() == unsat:
            return None
        
        return Region.from_model(self.solver.model(), self.vars)

    def block_region(self, region: Region):
        pass


if __name__ == "__main__":
    import random

    random.seed(105523)
    split = {
        i: sorted([random.uniform(0, 10) for j in range(3)])
        for i in range(3)
    }
    print('\n'.join([f"{i}: {split[i]}" for i in split.keys()]))

    program = ExplanationProgram(split)
    region = program.solve()
    print(region)