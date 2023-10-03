from pprint import pprint
from src.generators.rc2_generator import SeedGenerator

from pysat.examples.musx import MUSX


def test1():
    M = 3
    N = 3
    domains = {i: list(range(M)) for i in range(N)}
    g = SeedGenerator(domains)
    seed = g.get_seed()
    generated = [seed]
    while seed:
        print(seed)
        print(f"Volume: {}")
        g.block_up(seed)
        seed = g.get_seed()
        generated.append(seed)
    print("done")

if __name__ == "__main__":
    test1()