from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Iterator
from utils import *
import torch

# frozen to be hashable
@dataclass(frozen=True)
class Literal:
    variable: int
    negation: bool

    def __repr__(self):
        if self.negation:
            return '¬' + str(self.variable)
        else:
            return str(self.variable)

    def neg(self) -> 'Literal':
        """
        Return the negation of this literal.
        """
        return Literal(self.variable, not self.negation)


@dataclass
class Clause:
    # literals: List[Literal]

    def __init__(self, tensor):
        self.tensor = tensor

    def __repr__(self):
        # string = ''
        # for i in self.literals:
        #     string += str(i)
        # return string
        # return '∨'.join(map(str, self.literals))
        return "∨".join([str(i) if i>0 else "¬" + str(-i) for i in self.tensor if i!=0])

    def __iter__(self) -> Iterator[Literal]:
        return iter(self.literals)

    def __len__(self):
        return len(self.literals)

@dataclass
class Formula:
    # clauses: List[Clause]
    # __variables: Set[int]

    def __init__(self, clauses: torch.tensor):
        """
        Remove duplicate literals in clauses.
        """
        self.__variables = torch.unique(torch.abs(clauses))[1:] #pass
        self.clauses = clauses
        # self.clauses = []
        # self.__variables = set()
        # for clause in clauses:
        #     self.clauses.append(Clause(list(set(clause))))
        #     for lit in clause:
        #         var = lit.variable
        #         self.__variables.add(var)

    def variables(self) -> Set[int]:
        """
        Return the set of variables contained in this formula.
        """
        return self.__variables

    def __repr__(self):
        return ' ∧ '.join(f'({"∨".join([str(i) if i>0 else "¬" + str(-i) for i in clause if i!=0])})' for clause in self.clauses.tolist())

    def __iter__(self) -> Iterator[torch.tensor]:
        return iter(self.clauses)

    def __len__(self):
        return len(self.clauses)

@dataclass
class Assignment:
    value: bool
    antecedent: Optional[torch.tensor]
    dl: int  # decision level

class Assignments(dict):
    """
    The assignments, also stores the current decision level.
    """
    def __init__(self, max_len):
        super().__init__()
        '''
        self.assigns
        0: value
        1: antecedent
        2: dl
        3: is visited
        4: is in
        '''
        self.assigns = torch.zeros((max_len, 5))
        self.assigns[1, :] = -1
        # the decision level
        self.dl = 0

    def value(self, literal: int) -> bool:
        """
        Return the value of the literal with respect the current assignments.
        """
        if literal < 0:
            if self.assigns[(-literal)-1][3] == 0:
                self.assigns[(-literal)-1][3] = 1
                return None
            if self.assigns[(-literal)-1][0] == 0:
                return True
            else:
                return False
        else:
            if self.assigns[literal-1][3] == 0:
                self.assigns[literal-1][3] = 1
                return None
            if self.assigns[literal-1][0] == 0:
                return False
            else:
                return True

    def assign(self, variable, val, antecedent: int):
        # self[variable] = Assignment(value, antecedent, self.dl)
        val = variable if val else -variable
        self.assigns[val-1][0] = val
        self.assigns[val-1][1] = antecedent
        self.assigns[val-1][2] = self.dl
        self.assigns[val-1][4] = 1

    def unassign(self, variable: int):
        self.pop(variable)

    def satisfy(self, formula: Formula) -> bool:
        """
        Check whether the assignments actually satisfies the formula. 
        """
        for clause in formula:
            if True not in [self.value(lit) for lit in clause]:
                return False

        return True


