import re
from math import log
from itertools import combinations

from pysat.formula import WCNFPlus, IDPool
from pysat.card import CardEnc
from pysat.examples.rc2 import RC2Stratified

from src.regions import Region
from src.utils.sat_shortcuts import *


class SeedGenerator:
    """
    Generate unblocked seed with maximum volume.
    """
    def __init__(self, domains, solver="cd15"):
        self.domains = domains
        self.vpool = IDPool(start_from=1)
        self.wcnf = WCNFPlus()

        self._make_encoding()
        self.rc2 = RC2Stratified(self.wcnf, solver=solver)
    
    def _make_encoding(self):
        l, u, I = self._get_index_functions()

        # Upper and lower boolean variables
        for i in self.domains.keys():
            d = self.domains[i]
            for j in range(len(d)):
                l(i,j)  # l_ij <-> d[j] is lower bound
                u(i,j)  # u_ij <-> d[j] is upper bound
            for j in range(len(d)):
                if j == 0:
                    self.wcnf.append([-u(i,j)])
                elif j == len(d)-1:
                    self.wcnf.append([-l(i,j)])
                else:
                    l_lt_u = If(
                        l(i,j),
                        Not(Or([u(i,k) for k in range(j+1)]))
                    )  # l_ij -> ~(u_i0 v u_i1 v ... v u_ij)
                    self.wcnf.extend(l_lt_u.to_cnf())

                    u_gt_t = If(
                        u(i,j),
                        Not(Or([l(i,k) for k in range(j, len(d))]))
                    )  # u_ij -> ~(l_ij v l_i(j+1) v ... v l_im)
                    self.wcnf.extend(u_gt_t.to_cnf())
        
        # Interval boolean variables
        for i in self.domains.keys():
            d = self.domains[i]
            for (j, k) in combinations(range(len(d)), 2):
                I(i,j,k)  # I_ijk <-> interval is (d[j], d[k])
                constraint = Iff(And([l(i,j),u(i,k)]), I(i,j,k))
                self.wcnf.extend(constraint.to_cnf())  # (l_ij ^ u_ik) <-> I_ijk
                self.wcnf.append([I(i,j,k)], weight=log(d[k]-d[j])+1)
            I_vars = [I(i,j,k) for (j, k) in combinations(range(len(d)), 2)]
            card = CardEnc.equals(I_vars, vpool=self.vpool).clauses
            self.wcnf.extend(card)  # Exactly one I_ijk 

    def get_seed(self) -> Region:
        model = self.rc2.compute()
        if model is None:
            return None

        is_used_interval = lambda x: self.vpool.obj(x) and "I" in self.vpool.obj(x)
        intervals = [self.vpool.obj(x) for x in model if is_used_interval(x)]
        bounds = {}
        for I in intervals:
            f_id = int(I[1])
            l_idx = int(I[2])
            u_idx = int(I[3])
            d = self.domains[f_id]
            bounds[f_id] = (d[l_idx], d[u_idx])
        return Region(bounds)

    def must_contain(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        for i in d_idx.keys():
            d = self.domains[i]
            l_idx = d_idx[i][0]
            u_idx = d_idx[i][1]
            self._extend_rc2(Or([l(i,j) for j in range(l_idx+1)]).to_cnf())
            self._extend_rc2(Or([u(i,k) for k in range(u_idx, len(d))]).to_cnf())

    def block_up(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.domains[i]
            l_idx = d_idx[i][0]
            u_idx = d_idx[i][1]
            to_disjunct.append(Or([l(i,j) for j in range(l_idx+1, len(d))]))
            to_disjunct.append(Or([u(i,k) for k in range(u_idx)]))
        self._extend_rc2(Or(to_disjunct).to_cnf())

    def block_down(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = self._region_to_didx(r)
        to_disjunct = []
        for i in d_idx.keys():
            d = self.domains[i]
            l_idx = d_idx[i][0]
            u_idx = d_idx[i][1]
            to_disjunct.append(Or([l(i,j) for j in range(l_idx)]))
            to_disjunct.append(Or([u(i,k) for k in range(u_idx+1, len(d))]))
        self._extend_rc2(Or(to_disjunct).to_cnf())
    
    def _print_enc(self, s):
        s = str(s)
        d = {str(k): o for (k, o) in self.vpool.id2obj.items()}
        keys = (re.escape(k) for k in d.keys())
        pattern = re.compile(r'\b(' + '|'.join(keys) + r')\b')
        print(pattern.sub(lambda x: d[x.group()], s))
    
    def _extend_rc2(self, cnf):
        for clause in cnf:
            self.rc2.add_clause(clause)

    def _region_to_didx(self, r: Region):
        l, u, I = self._get_index_functions()
        d_idx = {}
        for i, b in r.bounds.items(): 
            d = self.domains[i]
            d_idx[i] = (d.index(b[0]), d.index(b[1]))
        return d_idx
    
    def _get_index_functions(self):
        l = lambda i, j: self.vpool.id(f"l{i}{j}")
        u = lambda i, j: self.vpool.id(f"u{i}{j}")
        I = lambda i, j, k: self.vpool.id(f"I{i}{j}{k}")
        return l, u, I