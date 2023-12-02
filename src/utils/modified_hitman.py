import collections

from pysat.examples.rc2 import RC2, RC2Stratified
from pysat.examples.lbx import LBX
from pysat.examples.mcsls import MCSls
from pysat.formula import WCNFPlus
from pysat.solvers import Solver, SolverNames
from pysat.examples.hitman import Hitman


class ModHitman(Hitman):
    def init(self, bootstrap_with, weights=None, subject_to=[]):
        # formula encoding the sets to hit
        formula = WCNFPlus()

        # hard clauses
        for to_hit in bootstrap_with:
            to_hit = map(lambda obj: self.idpool.id(obj), to_hit)

            formula.append([self.phase * vid for vid in to_hit])

        # soft clauses
        for obj_id in self.idpool.id2obj.keys():
            formula.append([-obj_id],
                    weight=1 if not weights else weights[self.idpool.obj(obj_id)])

        # additional hard constraints
        for cl in subject_to:
            if not len(cl) == 2 or not type(cl[0]) in (list, tuple, set):
                # this is a pure clause
                formula.append(list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), cl)))
            else:
                # this is a native AtMostK constraint
                formula.append([list(map(lambda a: self.idpool.id(a.obj) * (2 * a.sign - 1), cl[0])), cl[1]], is_atmost=True)

        if self.htype == 'rc2':
            if not weights or min(weights.values()) == max(weights.values()):
                self.oracle = RC2(formula, solver=self.solver, adapt=self.adapt,
                        exhaust=self.exhaust, minz=self.minz, trim=self.trim)
            else:
                self.oracle = RC2Stratified(formula, solver=self.solver,
                        adapt=self.adapt, exhaust=self.exhaust, minz=self.minz,
                        nohard=True, trim=self.trim)
        elif self.htype == 'lbx':
            self.oracle = LBX(formula, solver_name=self.solver,
                    use_cld=self.usecld)
        elif self.htype == 'mcsls':
            self.oracle = MCSls(formula, solver_name=self.solver,
                    use_cld=self.usecld)
        else:  # 'sat'
            assert self.solver in SolverNames.minisatgh + \
                    SolverNames.lingeling + SolverNames.cadical153, \
                    'Hard polarity setting is unsupported by {0}'.format(self.solver)

            assert formula.atms == [], 'Native AtMostK constraints aren\'t' \
            'supported by MinisatGH, Lingeling, or CaDiCaL 153'

            # setting up a SAT solver, so that it supports the same interface
            self.oracle = Solver(name=self.solver, bootstrap_with=formula.hard,
                                 use_timer=True)

            # MinisatGH supports warm start mode
            if self.solver in SolverNames.minisatgh:
                self.oracle.start_mode(warm=True)

            # soft clauses are enforced by means of setting polarities
            self.oracle.set_phases(literals=[self.phase * cl[0] for cl in formula.soft])

            # "adding" the missing compute() and oracle_time() methods
            self.oracle.compute = lambda: [self.oracle.solve(), self.oracle.get_model()][-1]
            self.oracle.oracle_time = self.oracle.time_accum

            # adding a dummy VariableMap, as is in RC2 and LBX/MCSls
            VariableMap = collections.namedtuple('VariableMap', ['e2i', 'i2e'])
            self.oracle.vmap = VariableMap(e2i={}, i2e={})
            for vid in self.idpool.id2obj.keys():
                self.oracle.vmap.e2i[vid] = vid
                self.oracle.vmap.i2e[vid] = vid
