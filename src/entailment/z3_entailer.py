from z3 import *

from ..model import Model
from ..regions import Region


class EntailmentChecker:
    _no_parent = 2147483647

    def __init__(self, model: Model):
        self.model = model
        self.grp_vars = {grp_id : [] for grp_id in set(self.model.tree_info)}
        self.used_features = model.thresholds.keys()
        self.feature_vars = {
            i: Real('x%d' % i) 
            for i in range(self.model.num_feature)
        }
        self.constraints = []
        self.cexample = None
        self.oracle_calls = 0

        self._encode_model()

    def __str__(self):
        return str(self.feature_vars) + str(self.grp_vars) + str(self.constraints)
    
    def predict(self, x: list[float], ws=None):
        if ws is None:
            ws = self._get_weights(x)
        elif not type(ws) == list:
            ws = list(ws)

        objective = self.model.objective
        if objective == "binary:logistic":
            return 0 if ws[0] < 0 else 1
        elif objective == "multi:softprob" or objective == "multi:softmax":
            return ws.index(max(ws))

    def entails(self, r: Region, out):
        objective = self.model.objective
        if out not in self.grp_vars.keys() and "multi" in objective:
            raise ValueError(f"{out} not in valid classes {self.grp_vars.keys()}")

        if objective == "binary:logistic":
            w = Sum(self.grp_vars[0])
            objective_c = w > 0 if out == 0 else w < 0
            if self._exists_counterexample(r, objective_c):
                return False
        elif objective == "multi:softprob" or objective == "multi:softmax":
            for grp in self.grp_vars.keys():
                if grp == out:
                    continue
                objective_c = Sum(self.grp_vars[out]) < Sum(self.grp_vars[grp])
                if self._exists_counterexample(r, objective_c):
                    return False
        else:
            raise NotImplementedError(f"objective {objective} not implemented")
        return True
    
    def reset(self):
        self.cexample = None
        self.oracle_calls = 0
    
    def _get_weights(self, x: list[float]):
        x_enc = And([
            self.feature_vars[i] == x[i] 
            for i in range(self.model.num_feature)
        ])

        solver = Solver()
        solver.add(*self.constraints, x_enc)
        if solver.check() == unsat:
            raise ValueError("error: unsat prediction")
        self.oracle_calls += 1

        model = solver.model()
        ws = [-1 for i in range(max(self.grp_vars.keys())+1)]
        for grp in self.grp_vars.keys():
            ws[grp] = sum([
                float(model[w_var].as_decimal(30))
                for w_var in self.grp_vars[grp]
            ])
        return ws 

    def _encode_model(self):
        for tree in self.model.trees:
            grp_id = self.model.tree_info[tree.tree_id]
            w_var = Real('w%d' % tree.tree_id)
            self.grp_vars[grp_id].append(w_var)
            for node_id in range(len(tree.nodes)):
                if tree.is_leaf(node_id) and not tree.is_deleted(node_id):
                    path_enc = self._encode_path(tree, node_id)
                    leaf_w = tree.split_condition(node_id)
                    self.constraints.append(Implies(path_enc, w_var == leaf_w))

    def _encode_path(self, tree, node_id):
        path = []
        while tree.parent(node_id) != self._no_parent:
            parent_id = tree.parent(node_id)
            split_ind = tree.split_index(parent_id)
            split_val = tree.split_condition(parent_id)
            if tree.left_child(parent_id) == node_id:
                path.append(self.feature_vars[split_ind] < split_val)
            elif tree.right_child(parent_id) == node_id:
                path.append(self.feature_vars[split_ind] >= split_val)
            node_id = parent_id 
        return And(*path)

    def _exists_counterexample(self, r: Region, objective):
        r_enc = And([
            And(
                self.feature_vars[f_id] >= r.bounds[f_id][0],
                self.feature_vars[f_id] < r.bounds[f_id][1]
            )
            for f_id in r.bounds.keys() 
        ])

        solver = Solver()
        solver.add(*self.constraints, objective, r_enc)
        self.oracle_calls += 1
        if solver.check() == unsat:
            self.cexample = None
            return False
        else:
            self.cexample = []
            for f_id in self.feature_vars.keys():
                result = solver.model()[self.feature_vars[f_id]]
                if result is not None:
                    result = float(result.as_decimal(30))
                self.cexample.append(result)
            return True