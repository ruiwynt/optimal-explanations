import bisect
from z3 import *

from entailment import EntailmentChecker
from regions import Region, FeatureSpaceInfo, FeatureVariables
from json_parser import Model


class ExplanationProgram:
    # uping indexes
    _f_id = 0
    _threshold_i = 1
    _bound_type = 2

    def __init__(self, model: Model, domains=None):
        self.model = model
        self.entailment_checker = EntailmentChecker(model)
        self.n_features = model.num_feature
        self.fs_info = FeatureSpaceInfo(model.thresholds)
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
    
    def _instance_to_region(self, instance: list[float]) -> Region:
        bounds = {}
        for i in range(len(instance)): 
            thresholds = self.fs_info.thresholds[i]
            bound_i = bisect.bisect_left(thresholds, instance[i])
            if bound_i == len(thresholds):
                bounds[i] = (thresholds[bound_i-2], thresholds[bound_i-1])
            elif bound_i == 0:
                bounds[i] = (thresholds[bound_i], thresholds[bound_i+1])
            else:
                bounds[i] = (thresholds[bound_i-1], thresholds[bound_i])
        return Region(bounds)
    
    def explain(self, instance: list[float], c, block_score=False):
        """Find all maximal explanations which contain the instance."""
        region = self._instance_to_region(instance)
        self.solver.add(
            And([
                And(
                    self.vars[i].lower <= region.bounds[i][0], 
                    self.vars[i].upper >= region.bounds[i][1]
                )
                for i in self.vars.keys()
            ])
        )
        self.enumerate_explanations(c, block_score=block_score)
    
    def enumerate_explanations(self, c, block_score=False):
        """Enumerate all maximal explanations for the given class."""
        region = self.get_region()
        while region:
            if not self.entailment_checker.entails(region, c):
                self._step_region(region, c, "down")
                self._block_up(region)
            else:
                self._step_region(region, c, "up")
                self._block_down(region)
                if block_score:
                    self.solver.add(self._var_score() > self.get_score(region))
                self._print_region(region)
            region = self.get_region()
        self._print_region(self.region)
    
    def get_region(self) -> Region:
        if self.solver.check() == unsat:
            return None
        
        self.region = Region.from_model(self.solver.model(), self.vars)
        return self.region
    
    def _print_region(self, region):
        print(f"-- ENTAILS --")
        print(region)
        print(f"Score: {self.get_score(region)}")
        print("-------------")

    def _block_down(self, region: Region):
        """Block all regions which are contained within the given region."""
        self.blocked_down.append(region)

        self.solver.add(
            Or([
                Or(self.vars[i].lower < region.bounds[i][0], self.vars[i].upper > region.bounds[i][1])
                for i in self.vars.keys()
            ])
        )

    def _block_up(self, region: Region):
        """Block all regions which contain the given region."""
        self.blocked_up.append(region)

        self.solver.add(
            Or([
                Or(self.vars[i].lower > region.bounds[i][0], self.vars[i].upper < region.bounds[i][1])
                for i in self.vars.keys()
            ])
        )
    
    def _step_region(self, region: Region, c: str, mode: str):
        if mode not in ("up", "down"):
            raise ValueError(f"error: invalid mode {mode}")
        queue = [
            (
                f_id,
                bisect.bisect_left(
                    self.fs_info.thresholds[f_id], 
                    region.bounds[f_id][side]
                ),
                side
            )
            for side in (0, 1) for f_id in region.bounds.keys()
        ]

        while len(queue) > 0:
            step = queue.pop(0)
            f_id = step[0]
            threshold_i = step[1]
            bound_type = step[2]
            t_list = self.fs_info.thresholds[f_id]

            # Cannot expand bounds any further
            if threshold_i <= 0 or threshold_i >= len(t_list)-1:
                continue

            # Cannot down bounds any further
            if bound_type == 1 and t_list[threshold_i-1] == region.bounds[f_id][0] or \
                bound_type == 0 and t_list[threshold_i+1] == region.bounds[f_id][1]:
                continue

            original_bound = region.bounds[f_id][:]  # Copy
            if bound_type == 0:
                i_step = 1 if mode == "down" else -1
                region.bounds[f_id] = (
                    t_list[threshold_i+i_step],
                    region.bounds[f_id][1]
                )
            else:
                i_step = 1 if mode == "up" else -1
                region.bounds[f_id] = (
                    region.bounds[f_id][0],
                    t_list[threshold_i+i_step]
                )
            
            entails = self.entailment_checker.entails(region, c)
            if entails and mode == "down" or not entails and mode == "up":
                region.bounds[f_id] = original_bound
            else:
                threshold_i = threshold_i + i_step
                queue.append(
                    (f_id, threshold_i, bound_type)
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
    
    def reset(self):
        pass
    

if __name__ == "__main__":
    import random
    import json

    with open("model.json", "r") as f:
        model = Model(json.loads(f.read()))
    # print('\n'.join([f"{i}: {model.thresholds[i]}" for i in model.thresholds.keys()]))

    instance = [7.4, 2.8, 6.1, 1.9]
    program = ExplanationProgram(model)
    program.explain(instance, 2, block_score=False)