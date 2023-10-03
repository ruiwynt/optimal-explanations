from src.traverser import LatticeTraverser
from src.regions import Region

def test():
    N = 5
    thresholds = {i: [j for j in range(10)] for i in range(N)}
    r = Region({i: (0, 9) for i in range(N)})

    class TempEntailer:
        def entails(self, r, c):
            for f_id, b in r.bounds.items():
                if b[0] > 3 or b[1] < 7:
                    return True
            return False
    
    stepper = LatticeTraverser(TempEntailer(), thresholds)
    stepper.shrink(r, "1")
    