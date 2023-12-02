import re
from math import log
from itertools import combinations
from decimal import Decimal

from pysat.formula import IDPool
from pysat.card import CardEnc
from pysat.examples.hitman import Atom

from src.regions import Region
from src.utils.sat_shortcuts import *
from src.utils.modified_hitman import ModHitman


class SeedGenerator:
    """
    Generate unblocked seed with maximum volume.
    """
    def __init__(self, fs_info, solver="g4"):
        self.fs_info = fs_info
        self.vpool = IDPool(start_from=1)
        self.hard = []
        self.to_hit = None
        self.weights = {}
        self.solver = solver
        self.interval_sizes = {}
        self.constraints = []

        self._init_hard_bounds()
        self._init_hard_intervals()
        self._init_soft()

        self.hitman = ModHitman(
            bootstrap_with=self.to_hit,
            weights=self.weights,
            subject_to=self._to_atoms(self.hard),
            solver="g4",
            htype="rc2",
            mxs_adapt=True,
            mxs_exhaust=True,
            mxs_minz=True,
            mcs_usecld=True,
        )

        self.n_vars = len(self.vpool.obj2id.keys())
        self.n_clauses = len(self.hard) + len(self.to_hit)
        # self._print_constraints()

    def _to_atoms(self, clauses):
        return [[Atom(abs(x), sign=True if x >= 0 else False) for x in c] for c in clauses]
    
    def _init_hard_bounds(self):
        l, u, I = self._get_index_functions()
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            for j in range(len(d)):
                l(i,j)  # l_ij <-> d[j] is lower bound
                u(i,j)  # u_ij <-> d[j] is upper bound
            self.hard.append([-l(i,len(d)-1)])  # Lower bound can't be highest threshold
            self.constraints.append(f"~{l(i,len(d)-1)}")
            self.hard.append([-u(i,0)])  # Upper bound can't be lowest threshold
            self.constraints.append(f"~{u(i,0)}")
            for j in range(1, len(d)-1):
                l_lt_u = Implies(
                    l(i,j),
                    Not(Or([u(i,k) for k in range(j+1)]))
                )  # l_ij -> ~(u_i0 v u_i1 v ... v u_ij)
                self.constraints.append(l_lt_u)
                self.hard += l_lt_u.to_cnf()

                u_gt_t = Implies(
                    u(i,j),
                    Not(Or([l(i,k) for k in range(j, len(d))]))
                )  # u_ij -> ~(l_ij v l_i(j+1) v ... v l_im)
                self.constraints.append(u_gt_t)
                self.hard += u_gt_t.to_cnf()
        
    def _init_hard_intervals(self):
        l, u, I = self._get_index_functions()
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            sizes = []
            for (j, k) in combinations(range(len(d)), 2):
                I(i,j,k)  # I_ijk <-> interval is (d[j], d[k])
                constraint = Iff(And([l(i,j),u(i,k)]), I(i,j,k))
                self.constraints.append(constraint)
                self.hard += constraint.to_cnf()  # (l_ij ^ u_ik) <-> I_ijk
                sizes.append(d[k]-d[j])
            self.interval_sizes[i] = sorted(sizes, reverse=True)

            l_vars = [l(i,j) for j in range(len(d))]
            u_vars = [u(i,k) for k in range(len(d))]
            self.constraints.append(f"sum({l_vars}) = 1")
            self.constraints.append(f"sum({u_vars}) = 1")
            for b_vars in (l_vars, u_vars):
                card = CardEnc.equals(b_vars, vpool=self.vpool).clauses
                self.hard += card  # Exactly one I_ijk 

    def _init_soft(self):
        l, u, I = self._get_index_functions()
        def w(i, j, k, factor):
            d = self.fs_info.get_domain(i)
            i_size = Decimal(d[k])-Decimal(d[j])
            return log(i_size) + log(factor)

        factor = 1
        all_intervals = [interval for f_intervals in self.interval_sizes.values() for interval in f_intervals]
        soft = {}
        while 1/factor in all_intervals:
            factor += 1
        for i in self.fs_info.keys():
            d = self.fs_info.get_domain(i)
            soft[i] = []
            for (j, k) in combinations(range(len(d)), 2):
                soft[i].append(I(i,j,k))
                self.weights[I(i,j,k)] = -w(i,j,k, factor)  # -ve so ModHitman maximises hs weight
        self.to_hit = soft.values()

    def get_seed(self) -> Region:
        model = self.hitman.get()
        if model is None:
            return None
        is_used_interval = lambda x: self.vpool.obj(x) and "I" in self.vpool.obj(x)
        intervals = [self.vpool.obj(x) for x in model if is_used_interval(x)]
        # self._print_enc(intervals)
        bounds = {}
        for I in intervals:
            I = I.split("_")
            f_id = int(I[1])
            l_idx = int(I[2])
            u_idx = int(I[3])
            d = self.fs_info.get_domain(f_id)
            bounds[f_id] = (d[l_idx], d[u_idx])
        return Region(bounds)

    def must_contain(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_conjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            u_idx = d_idx[i][1]
            to_conjunct.append(Or([l(i,j) for j in range(l_idx+1)]))
            to_conjunct.append(Or([u(i,k) for k in range(u_idx, len(d))]))
        self.constraints.append(And(to_conjunct))
        self._extend_hitman(And(to_conjunct).to_cnf())

    def block_up(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            if l_idx < len(d)-1:
                to_disjunct.append(Or([l(i,j) for j in range(l_idx+1, len(d))]))
            u_idx = d_idx[i][1]
            if u_idx > 0:
                to_disjunct.append(Or([u(i,k) for k in range(u_idx)]))
        self.constraints.append(Or(to_disjunct))
        self._extend_hitman(Or(to_disjunct).to_cnf())

    def block_down(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.fs_info.get_domain(i)
            l_idx = d_idx[i][0]
            if l_idx > 0:
                to_disjunct.append(Or([l(i,j) for j in range(l_idx)]))
            u_idx = d_idx[i][1]
            if u_idx < len(d)-1:
                to_disjunct.append(Or([u(i,k) for k in range(u_idx+1, len(d))]))
        self.constraints.append(Or(to_disjunct))
        self._extend_hitman(Or(to_disjunct).to_cnf())
    
    def _print_constraints(self):
        for c in self.constraints:
            self._print_enc(c)
    
    def _print_enc(self, s):
        s = str(s)
        d = {str(k): o for (k, o) in self.vpool.id2obj.items()}
        keys = (re.escape(k) for k in d.keys())
        pattern = re.compile(r'\b(' + '|'.join(keys) + r')\b')
        print(pattern.sub(lambda x: d[x.group()], s))
    
    def _extend_hitman(self, cnf):
        for clause in self._to_atoms(cnf):
            self.hitman.add_hard(clause)

    def _region_to_didx(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = {}
        for i, b in r.bounds.items(): 
            d = self.fs_info.get_domain(i)
            d_idx[i] = (d.index(b[0]), d.index(b[1]))
        return d_idx
    
    def _get_index_functions(self):
        l = lambda i, j: self.vpool.id(f"l_{i}_{j}")
        u = lambda i, j: self.vpool.id(f"u_{i}_{j}")
        I = lambda i, j, k: self.vpool.id(f"I_{i}_{j}_{k}")
        return l, u, I
