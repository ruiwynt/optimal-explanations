import heapq
import numpy as np
from math import log
from decimal import Decimal
from itertools import combinations

from ..regions import Region
from ..utils.np_regionlist import NpRegionList


class SeedGenerator:
    def __init__(self, fs_info):
        self.active_features = np.array(list(fs_info.active_features))
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
        self.blocked_down = None
        self.blocked_up = None

    def get_seed(self):
        r = self._get_seed()
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
        return self._ridx_to_r(best_ridxs)

    def _ridx_to_r(self, ridx):
        return Region({
            f_id: (self.pairs[f_id][pair_i][1], self.pairs[f_id][pair_i][2])
            for f_id, pair_i in ridx.items()
        })
    
    def must_contain(self, r):
        self.instance = r.to_numpy(self.active_features)
    
    def block_up(self, r):
        r = r.to_numpy(self.active_features)
        if self.blocked_up is None:
            self.blocked_up = NpRegionList(r.shape)
        self.blocked_up.add(r)

    def block_down(self, r):
        r = r.to_numpy(self.active_features)
        if self.blocked_down is None:
            self.blocked_down = NpRegionList(r.shape)
        self.blocked_down.add(r)

    def _heapscore(self, r_idxs):
        return -sum([
            Decimal(log(self.pairs[f_id][pair_i][0]))
            for f_id, pair_i in r_idxs.items()
        ])
    
    def _blocked(self, r):
        """Checks whether or not a region is blocked via numpy vectorisation"""
        r = r.to_numpy()
        if self.instance is not None:
            contains = np.zeros_like(self.instance)
            contains[:,0] = r[:,0] <= self.instance[:,0]
            contains[:,1] = r[:,1] >= self.instance[:,1]
            contains_instance = np.all(contains)
            if not contains_instance:
                return True
        r_lb = r[:,0]
        r_ub = r[:,1]
        if self.blocked_up is not None and len(self.blocked_up) > 0:
            bu_lb = self.blocked_up.data[:self.blocked_up.size,:,0]
            bu_ub = self.blocked_up.data[:self.blocked_up.size,:,1]
            blocked_up = np.zeros_like(self.blocked_up.data[:self.blocked_up.size,:,:])
            blocked_up[:,:,0] = np.logical_or(r_lb <= bu_lb, bu_lb == -1)
            blocked_up[:,:,1] = np.logical_or(r_ub >= bu_ub, bu_ub == -1)
            if np.any(np.all(blocked_up, axis=(1,2))):
                return True
        if self.blocked_down is not None and len(self.blocked_down) > 0:
            bd_lb = self.blocked_down.data[:self.blocked_down.size,:,0]
            bd_ub = self.blocked_down.data[:self.blocked_down.size,:,1]
            blocked_down = np.zeros_like(self.blocked_down.data[:self.blocked_down.size,:,:])
            blocked_down[:,:,0] = np.logical_or(r_lb >= bd_lb, bd_lb == -1)
            blocked_down[:,:,1] = np.logical_or(r_ub <= bd_ub, bd_ub == -1)
            if np.any(np.all(blocked_down, axis=(1,2))):
                return True
        return False
