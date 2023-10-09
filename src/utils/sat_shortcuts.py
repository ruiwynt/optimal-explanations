class Formula:
    def __init__(self):
        pass 

class Not(Formula):
    def __init__(self, formula):
        if formula == []:
            raise ValueError("empty list passed into formula")
        self.formula = formula
    
    def __str__(self):
        if type(self.formula) == int:
            return f"~{self.formula}"
        elif type(self.formula) == list:
            return f"~({str(And(self.formula))})"
        else:
            return f"~({str(self.formula)})"
    
    def to_cnf(self):
        if type(self.formula) == int:
            return [[-self.formula]]
        elif type(self.formula) == list:
            return And([-x for x in self.formula]).to_cnf()
        cnf = []
        for f in self.formula.to_cnf():
            cnf.append(And([-x for x in f]))
        return Or(cnf).to_cnf()

class Or(Formula):
    def __init__(self, formulas):
        if formulas == []:
            raise ValueError("empty list passed into formula")
        self.formulas = formulas
    
    def __str__(self):
        s = "("
        for f in self.formulas:
            s += f"{str(f)} v "
        s = s[:-3] + ")"
        return s

    def to_cnf(self):
        cnf = []
        f = self.formulas[0]
        if type(f) == int:
            cnf.append([f])
        elif type(f) == list:
            cnf.append(f)
        else:
            cnf = f.to_cnf()
        for f in self.formulas[1:]:
            if type(f) == int:
                cnf = [A + [f] for A in cnf]
            elif type(f) == list:
                cnf = [A + f for A in cnf]
            else:
                cnf = [A + B for A in f.to_cnf() for B in cnf]
        return cnf

class And(Formula):
    def __init__(self, formulas):
        if formulas == []:
            raise ValueError("empty list passed into formula")
        self.formulas = formulas

    def __str__(self):
        s = "("
        for f in self.formulas:
            s += f"{str(f)} ^ "
        s = s[:-3] + ")"
        return s
    
    def to_cnf(self):
        # TODO: Variable order consistency with Or
        enc = []
        for f in self.formulas:
            if type(f) == int:
                enc.append([f])
            elif type(f) == list:
                enc.append(f)
            else:
                enc += f.to_cnf()
        return enc

class Implies(Formula):
    def __init__(self, p, q):
        if p == [] or q == []:
            raise ValueError("empty list passed into formula")
        self.p = self._check_arg(p)
        self.q = self._check_arg(q)

    def __str__(self):
        return f"(({str(self.p)}) -> ({str(self.q)}))"
    
    def _check_arg(self, p):
        if type(p) == int:
            return Or([p])
        elif type(p) == list and len(p) > 1:
            print("List input to implies automatically interpreted as clause")
            return Or(p)
        else:
            return p
    
    def to_cnf(self):
        return Or([Not(self.p), self.q]).to_cnf()

class Iff(Formula):
    def __init__(self, p, q):
        if p == [] or q == []:
            raise ValueError("empty list passed into formula")
        self.p = self._check_arg(p)
        self.q = self._check_arg(q)

    def __str__(self):
        return f"(({str(self.p)}) <-> ({str(self.q)}))"

    def _check_arg(self, p):
        if type(p) == int:
            return Or([p])
        elif type(p) == list and len(p) > 1:
            print("List input to implies automatically interpreted as clause")
            return Or(p)
        else:
            return p
    
    def to_cnf(self):
        return And([Implies(self.p, self.q), Implies(self.q, self.p)]).to_cnf()

class EqualsOne(Formula):
    def __init__(self, literals):
        if not type(literals) == list:
            raise ValueError("EqualsOne only implemented for lists of literals")
        self.literals = literals
    
    def __str__(self):
        s = ""
        for x in self.literals:
            s += f"{x} + "
        return s[:-2] + "= one"
    
    def to_cnf(self):
        l = self.literals
        to_disjunct = []
        for x in self.literals:
            c = [-a for a in self.literals]
            c[c.index(-x)] = x
            to_disjunct.append(And(c))
        cnf = And([Or(self.literals), Or(to_disjunct)]).to_cnf()
        cleaned_cnf = []
        for clause in cnf:
            seen = set()
            skip = False
            for x in clause:
                if -x in seen:
                    skip = True
                    break
                else:
                    seen.add(x)
            if not skip:
                cleaned_cnf.append(list(set(clause)))
        return cleaned_cnf