import sys
import random
from pprint import pprint

from typing import List, Set, Tuple, Optional, Iterator
from classes import *
import torch

def cdcl_solve(formula: Formula) -> Optional[Assignments]:
    """
    Solve the CNF formula.

    If SAT, return the assignments.
    If UNSAT, return None.
    """
    assignments = Assignments(formula.variables().size(0))
    # First, do unit propagation to assign the initial unit clauses 
    reason, clause = unit_propagation(formula, assignments)
    #print(assignments.assigns)
    if reason == 'conflict':
        return None
    
    while not all_variables_assigned(formula, assignments):
        var, val = pick_branching_variable(formula, assignments)
        assignments.dl += 1
        # assignments.assign(var, val, antecedent=None)
        # I changed the antecedent value into -2
        assignments.assign(var, val, -2)
        while True:
            # print('before unit_propagation', assignments.assigns, formula.clauses.shape)
            reason, clause = unit_propagation(formula, assignments)
            #print('after unit_propagation', reason, clause, assignments.assigns)
            if reason != 'conflict':
                break
            b, learnt_clause = conflict_analysis(clause, assignments, formula)
            if b < 0:
                return None
            add_learnt_clause(formula, learnt_clause)
            backtrack(assignments, b)
            assignments.dl = b

            # The learnt clause must be a unit clause, so the
            # next step must again be unit progagation
    return assignments

def add_learnt_clause(formula: Formula, clause: Clause):
    new_clause = [len(formula.clauses)]
    new_clause.extend(clause.tolist())
    for i in range(len(new_clause), formula.clauses.shape[1]):
        new_clause.append(0)
    clause = torch.tensor(new_clause)
    formula.clauses = torch.cat((formula.clauses, clause.unsqueeze(0)))
    print(formula.clauses)

def all_variables_assigned(formula: Formula, assignments: Assignments) -> bool:
    return len(formula.variables()) == len(assignments)

def pick_branching_variable(formula: Formula, assignments: Assignments) -> Tuple[int, bool]:
    unassigned_vars = [var.item() for var in formula.variables() if assignments.assigns[var-1][4]==0]
    var = random.choice(unassigned_vars)
    val = random.choice([True, False])
    return (var, val)

def backtrack(assignments: Assignments, b: int):
    to_remove = []
    for var, assignment in enumerate(assignments.assigns):
        if assignment[2] > b:
            to_remove.append(var)
            print(to_remove)
    print("===move")
    print(to_remove)
    for var in to_remove:
        # assignments.pop(var)
        assignments.assigns[var][:5] = -1
        assignments.assigns[var][1] = -1
        #assignments.pop(var)

def clause_status(clause: Clause, assignments: Assignments) -> str:
    """
    Return the status of the clause with respect to the assignments.

    There are 4 possible status of a clause:
      1. Unit - All but one literal are assigned False
      2. Unsatisfied - All literals are assigned False
      3. Satisfied - All literals are assigned True
      4. Unresolved - Neither unit, satisfied nor unsatisfied
    """
    values = []
    for literal in clause:
        if literal < 0:
            if assignments.assigns[(-literal)-1][4] == 1:
                values.append(assignments.value(literal))
            else:
                values.append(None)
        elif literal > 0:
            if assignments.assigns[literal-1][4] == 1:
                values.append(assignments.value(literal))
            else:
                values.append(None)
    if True in values:
        return 'satisfied'
    elif values.count(False) == len(values):
        return 'unsatisfied'
    elif values.count(False) == len(values) - 1:
        return 'unit'
    else:
        return 'unresolved'


def unit_propagation(formula: Formula, assignments: Assignments) -> Tuple[str, Optional[Clause]]:
    # finish is set to True if no unit and conflict clause found in one iteration
    finish = False
    while not finish:
        finish = True
        for clause in formula:
            if len(clause) == formula.clauses.shape[1]:
                clause_t = clause[1:]
            status = clause_status(clause_t, assignments)
            if status == 'unresolved' or status == 'satisfied':
                continue
            elif status == 'unit':
                # select the literal to propagate
                
                for literal in clause_t:
                    if assignments.assigns[abs(literal)-1][4] == 0:
                        break

                # assign the variable according to unit rule
                assignments.assign(literal, literal>0, antecedent=clause[0])
                finish = False
            else:
                # conflict
                return ('conflict', clause)

    return ('unresolved', None)


def resolve(a: Clause, b: Clause, x: int) -> Clause:
    """
    The resolution operation
    """

    items = []
    for i in a:
        items.append(i.item())
    for i in b:
        items.append(i.item())
        
    result = set([i for i in items if -i not in items])

    # result = set(a.literals + b.literals) - {Literal(x, True), Literal(x, False)}
    result = torch.tensor(list(result))
    return result


def conflict_analysis(clause: torch.tensor, assignments: Assignments, formula: Formula) -> Tuple[int, Clause]:
    if assignments.dl == 0:
        return (-1, None)
 
    # literals with current decision level
    clause = clause[1:]
    literals = torch.tensor([literal for literal in clause if assignments.assigns[abs(literal)-1][2] == assignments.dl])

    #print(literals, assignments.dl)
    while len(literals) != 1:
        # Change the filter of antecedent into -2
        literals = filter(lambda lit: assignments.assigns[lit, 1] != -2, literals)
        literal = next(literals)
        antecedent = assignments.assigns[abs(literal)-1][1]
        clause = resolve(clause, formula.clauses[antecedent][1:], abs(literal))
        literals = torch.tensor([literal for literal in clause if assignments.assigns[abs(literal)-1][2] == assignments.dl])
    # out of the loop, `clause` is now the new learnt clause
    # compute the backtrack level b (second largest decision level)
    decision_levels = sorted(set(assignments.assigns[abs(literal)-1][2] for literal in clause))
    if len(decision_levels) <= 1:
        return 0, clause
    else:
        return decision_levels[-2], clause


def parse_dimacs_cnf(content: str) -> Formula:
    """
    parse the DIMACS cnf file format into corresponding Formula.
    """

    max_len = 0
    lines = 0
    for line in content.splitlines():
        tokens = line.split()
        if len(tokens) != 0 and tokens[0] not in ("p", "c"):
            lines += 1
            if max_len < len(tokens):
                max_len = len(tokens)
    clauses = torch.zeros((lines, max_len), dtype=int)

    row_num = 0
    for line in content.splitlines():
        tokens = line.split()
        if len(tokens) != 0 and tokens[0] not in ("p", "c"):
            for i, tok in enumerate(tokens):
                lit = int(tok)
                if lit == 0:
                    pass
                else:
                    clauses[row_num][i+1] = lit
            row_num += 1
    
    for i in range(row_num):
        clauses[i][0] = i
    return Formula(clauses)

