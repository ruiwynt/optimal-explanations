from src.utils.sat_shortcuts import *

def assert_equiv(A, B):
    for x in A:
        assert x in B
    for y in B:
        assert y in A

def test1():
    phi = Or(
        [
            And(
                [1, 2 ,3]
            ),
            And(
                [4, 5, 6]
            ),
            Not(
                And(
                    [7, 8, 9]
                )
            )
        ]
    ).to_cnf()

    phi2 = Or(
        [
            And(
                [1, 2 ,3]
            ),
            And(
                [4, 5, 6]
            ),
            Or(
                [-7, -8, -9]
            )
        ]
    ).to_cnf()

    assert_equiv(phi, phi2)
    
def test2():
    phi = If(
        And([1, 2, 3]),
        And([4, 5, 6])
    ).to_cnf()

    phi2 = Or(
        [
            Not(And([1, 2, 3])),
            And([4, 5, 6])
        ]
    ).to_cnf()

    assert_equiv(phi, phi2)

def test3():
    phi = If(
        And([1, 2, 3]),
        And([4, 5, 6])
    )

    phi2 = Or(
        [
            Not(And([1, 2, 3])),
            And([4, 5, 6])
        ]
    )

    phi = Not(phi).to_cnf()
    phi2 = Not(phi2).to_cnf()

    assert_equiv(phi, phi2)

def test4():
    phi = If(
        [1, 2, 3],
        4
    ).to_cnf()

    phi2 = If(
        Or([1, 2, 3]),
        Or([4])
    ).to_cnf()

    assert_equiv(phi, phi2)