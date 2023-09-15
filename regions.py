from z3 import *

from typing import Optional


DECIMAL_PREC = 99

class Region:
    def __init__(self, bounds: Optional[dict[int, tuple[float, float]]] = None):
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
            "%f < x%d <= %f" % (self.bounds[i][0], i, self.bounds[i][1]) 
            for i in sorted(self.bounds.keys())
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
                i: (min(thresholds[i])-100, max(thresholds[i])+100)
                for i in thresholds.keys()
            }
    
    def constraint_vals(self, i):
        return [self.domains[i][0]] + self.thresholds[i] + [self.domains[i][1]]
    
    def get_dmin(self, i):
        return self.domains[i][0]

    def get_dmax(self, i):
        return self.domains[i][1]