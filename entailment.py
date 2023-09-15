from z3 import *

from json_parser import Model
from regions import Region


class EntailmentChecker:
    _no_parent = 2147483647

    def __init__(self, model: Model):
        self.model = model
        self.grp_vars = {grp_id : [] for grp_id in set(self.model.tree_info)}
        self.feature_vars = {
            i: Real('x%d' % i) 
            for i in range(self.model.num_feature)
        }
        self.constraints = []
        self.cexample = None

        self._encode_model()

    def __str__(self):
        return str(self.feature_vars) + str(self.grp_vars) + str(self.constraints)
    
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

    def entails(self, region: Region, out):
        if out not in self.grp_vars.keys():
            print(f"{out} not in valid classes {self.grp_vars.keys()}")
            return
        
        objective = Or([
            Sum(self.grp_vars[grp]) > Sum(self.grp_vars[out])
            for grp in self.grp_vars.keys() if grp != out
        ])

        region_enc = And([
            And(
                self.feature_vars[f_id] >= region.bounds[f_id][0],
                self.feature_vars[f_id] < region.bounds[f_id][1]
            )
            for f_id in self.feature_vars.keys()
        ])

        solver = Solver()
        solver.add(*self.constraints, objective, region_enc)
        if solver.check() == unsat:
            self.cexample = None
            return True
        else:
            self.cexample = [
                solver.model()[self.feature_vars[f_id]]
                for f_id in self.feature_vars.keys()
            ]
            return False

if __name__ == "__main__":
    import json
    from xgboost import XGBClassifier

    with open("model.json", "r") as f:
        checker = EntailmentChecker(Model(json.loads(f.read())))
    
    out = 2
    region_bounds = {
        0: (-1, 8),  # Free
        1: (2, 4),  # Free
        2: (5.15, 7),
        3: (1.25, 1.55),
    }
    region = Region(region_bounds)
    if not checker.entails(region, out):
        print(checker.cexample)