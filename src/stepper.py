import logging
from math import isclose
from copy import deepcopy

from .entailment import EntailmentChecker
from .regions import Region


class RegionStepper:
    def __init__(
            self, 
            entailer: EntailmentChecker, 
            domains: dict[int: list[float]], 
            method="left"
        ):
        self.entailer = entailer 
        self.domains = domains
        self.method = method
        self.search_bounds = {
            f_id: (-1, len(domains[f_id])) 
            for f_id in self.domains.keys()
        }
    
    def must_contain(self, r: Region):
        self.search_bounds = {
            f_id: (self.domains[f_id].index(b[0]), self.domains[f_id].index(b[1]))
            for (f_id, b) in r.bounds.items()
        }

    def grow(self, r: Region, c: str):
        self._bsearch_step(r, c, "grow")

    def shrink(self, r: Region, c: str):
        self._bsearch_step(r, c, "shrink")

    def _bsearch_step(self, r: Region, c: str, mode: str):
        for (f_id, side) in ((i, j) for i in self.domains.keys() for j in (0, 1)):
            d = self.domains[f_id]
            bound = r.bounds[f_id]
            i = d.index(bound[0])
            j = d.index(bound[1])

            if side == 0 and mode == "grow":
                d = d[:i+1]
                d.reverse()
            elif side == 1 and mode == "grow":
                d = d[j:]
            elif side == 0 and mode == "shrink":
                j = min(self.search_bounds[f_id][0]+1, j)
                d = d[i:j]
            elif side == 1 and mode == "shrink":
                i = max(self.search_bounds[f_id][1], i+1)
                d = d[i:j+1]
                d.reverse()

            left = 0
            right = len(d)-1
            if left == right:
                continue
            while right - left > 1:
                mid = (left + right) // 2
                r.bounds[f_id] = (d[mid], bound[1]) if side == 0 else (bound[0], d[mid])

                entails = self.entailer.entails(r, c)
                if entails and mode == "shrink" or not entails and mode == "grow":
                    right = mid
                else:
                    left = mid 
            r.bounds[f_id] = (d[right], bound[1]) if side == 0 else (bound[0], d[right])
            entails = self.entailer.entails(r, c)
            if entails and mode == "shrink" or not entails and mode == "grow":
                r.bounds[f_id] = (d[left], bound[1]) if side == 0 else (bound[0], d[left])