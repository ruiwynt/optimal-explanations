import heapq
from math import log
from decimal import Decimal
from itertools import combinations

from ..regions import Region


class SeedGenerator:
    def __init__(self, fs_info):
        self.seen = set()
        self.pairs = {
            f_id: sorted([(c[1]-c[0], c[0], c[1]) for c in combinations(d, 2)], reverse=True)
            for f_id, d in fs_info.domains.items()
        }

        r_idxs = {f_id: 0 for f_id in self.pairs.keys()}
        self.seen.add(tuple(i for i in r_idxs.values()))
        self.ridxs_heap = [(self._heapscore(r_idxs), 0, r_idxs)]
        self.obj_id = 1

        self.instance = None
        self.blocked_down = []
        self.blocked_up = []

    def get_seed(self):
        r = self._get_seed()
        if self.instance is not None:
            while r is not None and self._blocked(r):
                r = self._get_seed()
        return r

    def _get_seed(self):
        if len(self.ridxs_heap) == 0:
            return None
        best_ridxs = heapq.heappop(self.ridxs_heap)[2]
        for f_id in self.pairs.keys():
            if best_ridxs[f_id] == len(self.pairs[f_id]) - 1:
                continue
            new_ridxs = best_ridxs.copy()
            new_ridxs[f_id] += 1
            r_enc = tuple(i for i in new_ridxs.values())
            if not r_enc in self.seen:
                self.seen.add(r_enc)
                heapq.heappush(
                    self.ridxs_heap, 
                    (self._heapscore(new_ridxs), self.obj_id, new_ridxs)
                )
                self.obj_id += 1
        bounds = {
            f_id: (self.pairs[f_id][pair_i][1], self.pairs[f_id][pair_i][2])
            for f_id, pair_i in best_ridxs.items()
        }
        return Region(bounds)
    
    def must_contain(self, r):
        self.instance = r
    
    def block_up(self, r):
        self.blocked_up.append(r)

    def block_down(self, r):
        self.blocked_down.append(r)

    def _heapscore(self, r_idxs):
        return -sum([
            Decimal(log(self.pairs[f_id][pair_i][0]))
            for f_id, pair_i in r_idxs.items()
        ])
    
    def _blocked(self, r):
        if not r.contains(self.instance):
            return True
        for b_reg in self.blocked_up:
            if r.blocked_up_by(b_reg):
                return True
        for b_reg in self.blocked_down:
            if r.blocked_down_by(b_reg):
                return True
        return False