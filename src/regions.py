from z3 import *

from typing import Optional
from math import isclose


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
            "%f <= x%d < %f" % (self.bounds[i][0], i, self.bounds[i][1]) 
            for i in sorted(self.bounds.keys())
        ]
        return "\n".join(region)
    
    def __eq__(self, r):
        for f_id in self.bounds.keys():
            if f_id not in r.bounds.keys():
                return False
        for f_id, b in r.bounds.items():
            if f_id not in self.bounds.keys():
                return False 
            if not isclose(b[0], self.bounds[f_id][0]) or \
                not isclose(b[1], self.bounds[f_id][1]):
                return False
        return True

    def contained_in(self, r):
        """True iff this region is contained within r"""
        if r.bounds == self.bounds:
            return True
        elif r.bounds == {}:
            return True
        elif self.bounds == {}:
            return False
        for f_id in self.bounds.keys():
            if not f_id in r.bounds.keys():
                continue
            b = self.bounds[f_id]
            rb = r.bounds[f_id]
            if b[0] < rb[0] or b[1] > rb[1]:
                return False
        return True
    
    def contains(self, r):
        """True iff this region contains r"""
        if r.bounds == self.bounds:
            return True
        elif self.bounds == {}:
            return True
        for f_id in self.bounds.keys():
            if not f_id in r.bounds.keys():
                return False
            b = self.bounds[f_id]
            rb = r.bounds[f_id]
            if b[0] > rb[0] or b[1] < rb[1]:
                return False
        return True
    
    def blocked_up_by(self, r):
        if r.bounds == {}:
            return True
        for f_id, rb in r.bounds.items():
            if not f_id in self.bounds.keys():
                continue
            b = self.bounds[f_id]
            if not (b[0] <= rb[0] and b[1] >= rb[1]):
                return False
        return True

    def blocked_down_by(self, r):
        if r.bounds == {}:
            return True
        for f_id, rb in r.bounds.items():
            if not f_id in self.bounds.keys():
                continue
            b = self.bounds[f_id]
            if not (b[0] >= rb[0] and b[1] <= rb[1]):
                return False
        return True
    
    @classmethod
    def from_z3model(cls, model, variables):
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


class LimitVariables:
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
    def __init__(self, thresholds: dict[int: list[float]], limits=None):
        """
        thresholds: {
                    <feature_id>: [<threshold value>, ...],
                    ...
                } 
        """
        self.thresholds = thresholds

        if limits:
            self.limits = limits
            for k in self.thresholds.keys():
                if not k in limits.keys():
                    raise KeyError(f"limits dict doesn't contain key {k}")
        else:
            self.limits = {
                i: (min(thresholds[i])-100, max(thresholds[i])+100)
                for i in thresholds.keys()
            }

        self.domains = {
            i: [self.limits[i][0]] + self.thresholds[i] + [self.limits[i][1]]
            for i in thresholds.keys()
        }
        for i, d in self.domains.items():
            if d[0] == d[1]:
                self.domains[i][0] -= 1
            if d[-1] == d[-2]:
                self.domains[i][-1] += 1
        
    def keys(self):
        return self.thresholds.keys()
    
    def get_domain(self, i):
        return self.domains[i]
    
    def get_dmin(self, i):
        return self.domains[i][-1]

    def get_dmax(self, i):
        return self.domains[i][0]