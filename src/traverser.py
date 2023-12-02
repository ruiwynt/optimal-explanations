from .entailment.z3_entailer import EntailmentChecker
from .regions import Region


class LatticeTraverser:
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
    
    def eliminate_vars(self, r: Region):
        to_remove = set()
        c = self.entailer.predict([
            (r.bounds[i][0] + r.bounds[i][1])/2 if i in r.bounds.keys() else -1 
            for i in range(self.entailer.model.num_feature)
        ])
        for f_id in r.bounds.keys():
            b = r.bounds[f_id]
            d = self.domains[f_id]
            r.bounds[f_id] = (d[0], d[len(d)-1])
            if not self.entailer.entails(r, c):
                r.bounds[f_id] = b
            else:
                to_remove.add(f_id)
        for f_id in to_remove:
            del r.bounds[f_id]

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
                d = d[i:j]
            elif side == 1 and mode == "shrink":
                d = d[i+1:j+1]
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