from dataclasses import dataclass
from typing import List, Set, Tuple, Optional, Iterator
from utils import *
import torch
import copy


@dataclass
class Formula:
    # clauses: List[Clause]
    # __variables: Set[int]

    def __init__(self, clauses: torch.tensor, row_num, max_len):
        """
        Remove duplicate literals in clauses.
        """
        self.clauses = []
        for i, clause in enumerate(clauses):
            if clause != []:
                uniq_clause = list(set(clause))
                uniq_clause.insert(0, i)
                for i in range(len(uniq_clause), max_len):
                    uniq_clause.append(0)
                self.clauses.append(uniq_clause)
        self.clauses = torch.tensor(self.clauses).cuda(0)
        self.__variables = torch.unique(torch.abs(self.clauses[:, 1:])) #pass
        self.__variables = self.__variables[self.__variables!=0]

    def variables(self) -> Set[int]:
        """
        Return the set of variables contained in this formula.
        """
        return self.__variables

    def __repr__(self):
        return ' ∧ '.join(f'({"∨".join([str(i) if i>0 else "¬" + str(-i) for i in clause if i!=0])})' for clause in self.clauses[:, 1:].tolist())

    def __iter__(self) -> Iterator[torch.tensor]:
        return iter(self.clauses)

    def __len__(self):
        return len(self.clauses)

# @dataclass
# class Assignment:
#     value: bool
#     antecedent: Optional[torch.tensor]
#     dl: int  # decision level

class Assignments():
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
        5: idx
        '''
        self.assigns = torch.zeros((max_len, 6), dtype=int).cuda(0)
        self.assigns[:, 1] = -1
        for i in range(max_len):
            self.assigns[i, 5] = i+1
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
        variable = abs(variable)
        self.assigns[variable-1][0] = val
        self.assigns[variable-1][1] = antecedent
        self.assigns[variable-1][2] = int(self.dl)
        self.assigns[variable-1][3] = 1
        self.assigns[variable-1][4] = 1

    def unassign(self, variable: int):
        self.pop(variable)

    def satisfy(self, formula: Formula) -> bool:
        """
        Check whether the assignments actually satisfies the formula. 
        """
        # for clause in formula:
        #     if True not in [self.value(lit) for lit in clause]:
        #         return False
        for i in formula:
            temp = i[1:]
            if True not in [self.value(lit) if lit !=0 else None for lit in temp]:
                return False

        return True


