from src.utils.sat_shortcuts import *

def assert_equiv(A, B):
    assert len(A) == len(B)
    for i in range(len(A)):
        A[i] = sorted(A[i])
        B[i] = sorted(B[i])
    A = sorted(A)
    B = sorted(B)
    for i in range(len(A)):
        assert A[i] == B[i]

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
    phi = Implies(
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
    phi = Implies(
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
    phi = Implies(
        [1, 2, 3],
        4
    ).to_cnf()

    phi2 = Implies(
        Or([1, 2, 3]),
        Or([4])
    ).to_cnf()

    assert_equiv(phi, phi2)

def test_Not():
    # Negate a variable
    assert_equiv(Not(1).to_cnf(), [[-1]])
    
    # Negate a clause
    assert_equiv(Not([1, 2]).to_cnf(), [[-1], [-2]])
    
    assert_equiv(Not(And([1, 2])).to_cnf(), [[-1, -2]])

def test_Or():
    # Disjunction of variables
    assert_equiv(Or([1, 2]).to_cnf(), [[1, 2]])
    
    # Disjunction of clauses
    assert_equiv(Or([[1, 2], [3, 4]]).to_cnf(), [[1, 2, 3, 4]])
    
    # Disjunction of formulas
    assert_equiv(Or([And([1, 2]), And([3, 4])]).to_cnf(), [[1, 3], [1, 4], [2, 3], [2, 4]])

def test_And():
    # Conjunction of variables
    assert_equiv(And([1, 2]).to_cnf(), [[1], [2]])
    
    # Conjunction of clauses
    assert_equiv(And([[1, 2], [3, 4]]).to_cnf(), [[1, 2], [3, 4]])
    
    # Conjunction of formulas
    assert_equiv(And([Or([1, 2]), Or([3, 4])]).to_cnf(), [[1, 2], [3, 4]])

def test_Implies():
    # Simple implication with variables
    assert_equiv(Implies(1, 2).to_cnf(), [[-1, 2]])
    
    # Compound implication with formulas
    assert_equiv(Implies(And([1, 2]), Or([3, 4])).to_cnf(), [[-1, -2, 3, 4]])

def test_Iff():
    # Simple biconditional with variables
    assert_equiv(Iff(1, 2).to_cnf(), [[-1, 2], [1, -2]])
    
    # Compound biconditional with formulas
    assert_equiv(Iff(And([1, 2]), Or([3, 4])).to_cnf(), [[-1, -2, 3, 4], [1, -3], [1, -4], [2, -3], [2, -4]])

def test_advanced_Not():
    # Negating an Or formula
    assert_equiv(Not(Or([1, 2])).to_cnf(), [[-1], [-2]])

    # Negating an Implies formula (1 -> 2)
    assert_equiv(Not(Implies(1, 2)).to_cnf(), [[1], [-2]])

    # Negating an Iff formula (1 <-> 2)
    assert_equiv(Not(Iff(1, 2)).to_cnf(), [[1, 2], [-1, -2], [-1, 1], [-2, 2]])

def test_advanced_Or():
    # Combining Not formulas
    assert_equiv(Or([Not(1), Not(2)]).to_cnf(), [[-1, -2]])

    # Combining various compound formulas (Or(1 And 2, 3 -> 4))
    assert_equiv(Or([And([1, 2]), Implies(3, 4)]).to_cnf(), [[1, -3, 4], [2, -3, 4]])

def test_advanced_And():
    # Combining Implies formulas (1 -> 2 And 3 -> 4)
    assert_equiv(And([Implies(1, 2), Implies(3, 4)]).to_cnf(), [[-1, 2], [-3, 4]])

    # Combining Iff formulas (1 <-> 2 And 3 <-> 4)
    assert_equiv(And([Iff(1, 2), Iff(3, 4)]).to_cnf(), [[-1, 2], [1, -2], [-3, 4], [3, -4]])

def test_advanced_Implies():
    # Complex implications with nested formulas (1 And (2 Or 3) -> 4 And 5)
    formula = Implies(And([1, Or([2, 3])]), And([4, 5]))
    assert_equiv(formula.to_cnf(), [[-1, -2, 4], [-1, -2, 5], [-1, -3, 4], [-1, -3, 5]])

def test_advanced_Iff():
    # Complex biconditionals with nested formulas (1 And 2 <-> 3 Or 4)
    formula = Iff(And([1, 2]), Or([3, 4]))
    assert_equiv(formula.to_cnf(), [[-1, -2, 3, 4], [1, -3], [1, -4], [2, -3], [2, -4]])