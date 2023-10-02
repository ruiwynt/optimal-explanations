import logging

from minizinc import Instance, Model, Solver

from ..regions import Region, FeatureSpaceInfo

class SeedGenerator:
    _SCORE_SCALE = 1e5

    def __init__(
            self, 
            fs_info: FeatureSpaceInfo, 
            score="volume", 
            solver="chuffed", 
            method="rand"
        ):
        self.fs_info = fs_info
        self.solver = Solver.lookup(solver)
        self.instance = Instance(self.solver)

        self.blocked_up = []
        self.blocked_down = []

        # Defining variables and domains.
        for f_id in fs_info.keys():
            dlen = len(fs_info.get_domain(f_id))
            self.instance.add_string(
                f"""
                var 0..{dlen-1}: x{f_id}l; 
                var 0..{dlen-1}: x{f_id}u; 
                constraint x{f_id}u > x{f_id}l;
                """
            )

    def get_seed(self) -> Region:
        result = self.instance.solve()
        if not result:
            return None
        bounds = {}
        for f_id in self.fs_info.keys():
            d = self.fs_info.get_domain(f_id)
            bounds[f_id] = (d[result[f"x{f_id}l"]], d[result[f"x{f_id}u"]])
        self.region = Region(bounds)
        for r in self.blocked_up:
            if self.region.contains(r):
                raise ValueError()
        for r in self.blocked_down:
            if self.region.contained_in(r):
                raise ValueError()
        return self.region

    def must_contain(self, r: Region):
        for f_id, b in r.bounds.items():
            d = self.fs_info.get_domain(f_id)
            self.instance.add_string(
                f"""
                constraint x{f_id}l <= {d.index(b[0])};
                constraint x{f_id}u >= {d.index(b[1])};
                """
            )

    def block_up(self, r: Region):
        c = "constraint "
        for f_id, b in r.bounds.items():
            d = self.fs_info.get_domain(f_id)
            c += f"(x{f_id}l > {d.index(b[0])}) \/ (x{f_id}u < {d.index(b[1])}) \/ "
        c = c[:-4] + ";\n"  # Remove final \/
        self.instance.add_string(c)
        self.blocked_up.append(r)

    def block_down(self, r: Region):
        c = "constraint "
        for f_id, b in r.bounds.items():
            d = self.fs_info.get_domain(f_id)
            c += f"(x{f_id}l < {d.index(b[0])}) \/ (x{f_id}u > {d.index(b[1])}) \/ "
        c = c[:-4] + ";\n"  # Remove final \/
        self.instance.add_string(c)
        self.blocked_down.append(r)