from dataclasses import dataclass
from typing import List, Set, Iterator
from utils import *
import torch

# frozen to be hashable
@dataclass(frozen=True)
class Literal:
    """
    This class represents a literal in propositional logic, which is a
    variable or its negation. The variable is an integer, and the
    negation is a boolean value.
    """
    variable: int
    negation: bool

    def __repr__(self):
        """
        Provides a string representation of the literal, suitable for
        debugging or logging.
        """
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
    """
    This class represents a clause in propositional logic, which is a
    disjunction of literals. The clause is represented as a list of
    literals.
    """
    literals: List[Literal]

    def __repr__(self):
        """
        Provides a string representation of a disjunction of literals,
        suitable for debugging or logging.
        """
        return '∨'.join(map(str, self.literals))

    def __iter__(self) -> Iterator[Literal]:
        """
        Allows the clause to be iterable, making it easy to iterate over
        its literals.
        """
        return iter(self.literals)

    def __len__(self):
        """
        Returns the number of literals in the clause.
        """
        return len(self.literals)

    def __hash__(self):
        """
        Returns a hash value for the clause, which is used to compare
        clauses for equality.
        """
        x = 0 
        for lit in self.literals:
            x ^= hash(lit)
        return x

class resolve_lit:
    """
    This class represents a literal in propositional logic, which is a
    variable or its negation. The variable is an integer, and the
    negation is a boolean value.
    """
    def __init__(self, val, hash_val):
        """
        Initializes the Literal object with a variable and a negation.
        """
        self.val = val
        self.hash_val = hash_val
    
    def __hash__(self):
        """
        Returns a hash value for the literal, which is used to compare
        literals for equality.
        """
        return self.hash_val

@dataclass
class Formula:
    """
    This class represents a logical formula in conjunctive normal form
    (CNF), where each formula consists of multiple clauses, and each 
    clause is a disjunction of literals.
    """

    def __init__(self, clauses: torch.tensor, max_len, args):
        """
        Initializes the Formula object by processing the input tensor 
        of clauses to remove duplicate literals and ensuring that all 
        clauses are of the same length by padding them.
        """
        self.clauses = []
        
        for i, clause in enumerate(clauses):
            if clause != []:
                uniq_clause = Clause(list(set(clause))).literals
                uniq_clause.insert(0, i)
                for j in range(1, len(uniq_clause)):
                    uniq_clause[j] = -uniq_clause[j].variable if uniq_clause[j].negation else uniq_clause[j].variable
                for j in range(len(uniq_clause), max_len):
                    uniq_clause.append(0)
                self.clauses.append(uniq_clause)
                
        if args.gpu:
            self.clauses = torch.tensor(self.clauses).cuda(0)
        else:
            self.clauses = torch.tensor(self.clauses)

        self.__variables = torch.unique(torch.abs(self.clauses[:, 1:])) #pass
        self.__variables = self.__variables[self.__variables!=0]

        self.max_len = self.__variables.shape[0]
        self.hash_val = torch.zeros(self.max_len*2, dtype=torch.long)
        for i, clause in enumerate(clauses):
            if clause != []:
                uniq_clause = Clause(list(set(clause))).literals
                for i in uniq_clause:
                    val = -i.variable if i.negation else i.variable
                    self.hash_val[self.abs_offset(val)] = hash(i)

    def variables(self) -> Set[int]:
        """
        Returns a set of all unique variables present in the formula.
        """
        return self.__variables

    def __repr__(self):
        """
        Provides a string representation of the formula, suitable for 
        debugging or logging.
        """
        return ' ∧ '.join(f'({"∨".join([str(i) if i>0 else "¬" + str(-i) for i in clause if i!=0])})' for clause in self.clauses[:, 1:].tolist())

    def __iter__(self) -> Iterator[torch.tensor]:
        """
        Allows the formula to be iterable, making it easy to iterate over 
        its clauses.
        """
        return iter(self.clauses)

    def __len__(self):
        """
        Returns the number of clauses in the formula.
        """
        return len(self.clauses)

    def abs_offset(self, a):
        """
        Converts a literal into an index position in a data structure used 
        to track clauses related to literals.
        """
        if a < 0:
            return abs(a)-1 + self.max_len
        else:
            return a-1


class Assignments():
    """
    This class manages a list of variable assignments for a SAT solver.
    It includes operations to assign, unassign, and evaluate literals 
    based on current assignments.
    """
    def __init__(self, max_len, args):
        """
        Initializes the Assignments object with space to track 
        values and metadata for each variable.
        """
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
        if args.gpu:
            self.assigns = torch.zeros((max_len, 6), dtype=int).cuda(0)
        else:
            self.assigns = torch.zeros((max_len, 6), dtype=int)
        self.assigns[:, 1] = -1
        for i in range(max_len):
            self.assigns[i, 5] = i+1
        # the decision level
        self.dl = 0

    def value(self, literal: int) -> bool:
        """
        Determines the value of a given literal based on current assignments.
        """
        if literal<0:
            return not self.assigns[abs(literal)-1][0] == 1
        else:
            return self.assigns[abs(literal)-1][0] == 1

    def assign(self, variable, val, antecedent: int):
        """
        Assigns a value to a variable, along with storing its antecedent 
        and the current decision level.
        """
        variable = abs(variable)
        self.assigns[variable-1][0] = val
        self.assigns[variable-1][1] = antecedent
        self.assigns[variable-1][2] = int(self.dl)
        self.assigns[variable-1][3] = 1
        self.assigns[variable-1][4] = 1

    def unassign(self, variable: int):
        """
        Removes an assignment for a variable, effectively making it unassigned.
        """
        self.assigns[abs(variable)-1]=0

    def satisfy(self, formula: Formula) -> bool:
        """
        Checks if the current assignments satisfy the given formula. 
        """
        for i in formula:
            temp = i[1:]
            if True not in [self.value(lit) if lit !=0 else None for lit in temp]:
                return False

        return True

