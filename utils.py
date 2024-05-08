import random
from typing import List, Set, Tuple, Optional, Iterator
from classes import *
import torch
import re

def abs_offset(a, tlength):
    """
    Converts a literal into an index position in a data structure used 
    to track clauses related to literals.
    """
    if a < 0:
        return abs(a)-1 + tlength
    else:
        return a-1

def init_watches(formula: Formula):
    """
    Initializes the watch list data structures used in the CDCL algorithm.
    
    Return lit2clauses and clause2lits
    """
    
    lit2clauses = torch.zeros(formula.variables().shape[0] * 2, formula.clauses.shape[0]+1, dtype=torch.int).to(formula.clauses.device)
    clause2lits = torch.zeros(formula.clauses.shape[0], 2, dtype=torch.int).to(formula.clauses.device)

    tlength = formula.variables().shape[0]
    lit2clauses[:, -1] = 1

    for idx, clause in enumerate(formula.clauses):
        temp_clause = clause[1:]
        try:
            if temp_clause[1] == 0:
                lit2clauses[abs_offset(temp_clause[0], tlength)][idx] = lit2clauses[abs_offset(temp_clause[0], tlength)][-1]
                lit2clauses[abs_offset(temp_clause[0], tlength)][-1] += 1
                clause2lits[idx][0] = temp_clause[0]
            else:
                lit2clauses[abs_offset(temp_clause[0], tlength)][idx] = lit2clauses[abs_offset(temp_clause[0], tlength)][-1]
                lit2clauses[abs_offset(temp_clause[0], tlength)][-1] += 1
                lit2clauses[abs_offset(temp_clause[1], tlength)][idx] = lit2clauses[abs_offset(temp_clause[1], tlength)][-1]
                lit2clauses[abs_offset(temp_clause[1], tlength)][-1] += 1
                clause2lits[idx][0] = temp_clause[0]
                clause2lits[idx][1] = temp_clause[1]
        except:
            pass
            
    return lit2clauses, clause2lits


def cdcl_solve(formula: Formula, args) -> Optional[Assignments]:
    """
    Main function to solve the SAT problem using the CDCL method.

    Return: An Assignments object with the solution if SAT; 
            None if UNSAT.
    """
    assignments = Assignments(formula.variables().size(0), args)
    lit2clauses, clause2lits = init_watches(formula)

    # First, do unit propagation to assign the initial unit clauses 
    unit_clauses = [clause for clause in formula if len(clause) > 2 and clause[2] == 0]
    to_propagate = []
    for clause in unit_clauses:
        lit = clause[1]
        var = lit
        val = lit>0
        if assignments.assigns[abs(var)-1, 4] == 0:
            assignments.assign(var, val, clause[0])
            to_propagate.append(lit)
    
    # reason, clause = unit_propagation(formula, assignments)
    reason, clause = unit_propagation(formula, assignments, lit2clauses, clause2lits, to_propagate)

    if reason == 'conflict':
        return None
    
    while not all_variables_assigned(formula, assignments):
        # val == True means negative value
        var, val = pick_branching_variable(formula, assignments)
        assignments.dl += 1
        assignments.assign(var, val, -1)
        to_propagate = [var if val else -var]
        while True:
            # reason, clause = unit_propagation(formula, assignments)
            reason, clause = unit_propagation(formula, assignments, lit2clauses, clause2lits, to_propagate)
            if clause != None:
                new_clause = [len(formula.clauses)]
                new_clause.extend(clause.tolist())
                for i in range(len(new_clause), formula.clauses.shape[1]):
                    new_clause.append(0)
                clause = torch.tensor(new_clause)
            
            if reason != 'conflict':
                break
            
            b, learnt_clause = conflict_analysis(clause, assignments, formula)
            if b < 0:
                return None
            
            lit2clauses, clause2lits, index = add_learnt_clause(formula, learnt_clause, assignments, lit2clauses, clause2lits)
            backtrack(assignments, b)
            assignments.dl = b

            # The learnt clause must be a unit clause, so the
            # next step must again be unit progagation
            for literal in learnt_clause:
                if assignments.assigns[abs(literal)-1, 4] == 0:
                    break
            assignments.assign(literal, literal>0, antecedent=index)
            to_propagate = [literal if literal < 0 else literal]
    return assignments.assigns

def add_learnt_clause(formula: Formula, clause, assignments, lit2clauses, clause2lits):
    """
    Adds a new learnt clause to the formula and updates related data structures.

    Return: Updated lit2clauses and clause2lits after adding the new clause.
    """
    total_len = formula.variables().shape[0]
    new_clause = [len(formula.clauses)]
    new_clause.extend(clause.tolist())
    for i in range(len(new_clause), formula.clauses.shape[1]):
        new_clause.append(0)
    clause = torch.tensor(new_clause)
    # extend current formula.clauses to save new larger clause
    if clause.shape[0] > formula.clauses.shape[1]:
        formula.clauses = torch.cat((formula.clauses, torch.zeros(formula.clauses.shape[0], clause.shape[0] - formula.clauses.shape[1], dtype=torch.int)), dim=1)
    formula.clauses = torch.cat((formula.clauses, clause.unsqueeze(0)))
    
    index = clause[0]
    clause_lit = clause[1:]
    clause_lit = clause_lit[clause_lit!=0]

    lit2clauses = torch.cat((lit2clauses, torch.zeros(lit2clauses.shape[0], 1)), dim=1).to(torch.int)
    clause2lits = torch.cat((clause2lits, torch.zeros(1,2))).to(torch.int)
    lit2clauses[:, -1] = lit2clauses[:, -2]
    lit2clauses[:, -2] = 0

    for lit in sorted(clause_lit, key=lambda lit: -assignments.assigns[abs(lit)-1, 2]):
        for i in range(len(clause2lits[index])):
            if clause2lits[index][i] == 0:
                clause2lits[index][i] = lit
                lit2clauses[abs_offset(lit, total_len)][index] = lit2clauses[abs_offset(lit, total_len)][-1]
                lit2clauses[abs_offset(lit, total_len)][-1] += 1
                break
        else:
            break
    return lit2clauses, clause2lits, clause2lits.shape[0]-1

def all_variables_assigned(formula: Formula, assignments: Assignments) -> bool:
    """
    Checks if all variables in the formula have been assigned a value.
    
    Return: Boolean indicating if all variables are assigned.
    """
    return len(formula.variables()) == len(assignments.assigns[assignments.assigns[:, 4] == 1])

def pick_branching_variable(formula: Formula, assignments: Assignments) -> Tuple[int, bool]:
    """
    Purpose: Selects the next variable to assign in the decision-making 
    process of CDCL.
    
    Return: A tuple (var, val) representing the variable to assign and 
    its Boolean value.
    """
    unassigned_vars = [var.item() for var in formula.variables() if assignments.assigns[var-1][4]==0]
    var = random.choice(unassigned_vars)
    val = random.choice([True, False])
    return (var, val)

def backtrack(assignments: Assignments, b: int):
    """
    Reverts assignments to a previous decision level during conflict resolution.
    """
    to_remove = []
    for var, assignment in enumerate(assignments.assigns):
        if assignment[2] > b:
            to_remove.append(var)
    for var in to_remove:
        assignments.assigns[var][:5] = 0
        assignments.assigns[var][1] = -1

def clause_status(clause, assignments: Assignments) -> str:
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


def unit_propagation(formula, assignments, lit2clauses, clause2lits, to_propagate):
    """
    Purpose: Propagates the implications of unit clauses throughout the formula.
    
    Return: A tuple with a status string ('conflict' or 'unresolved') and possibly 
    a conflicting clause.
    """
    total_len = formula.variables().shape[0]
    while len(to_propagate) > 0:
        watching_lit = -to_propagate.pop()
        watching_lit_idx = abs_offset(watching_lit, total_len)
        # use list(.) to copy it because size of 
        # lit2clauses[watching_lit]might change during for-loop
        x = torch.argsort(lit2clauses[watching_lit_idx, :-1][lit2clauses[watching_lit_idx, :-1]>0])
        watching_clauses = formula.clauses[lit2clauses[watching_lit_idx, :-1]>0]
        watching_clauses = watching_clauses[x]
        
        for watching_clause in watching_clauses:
            index = watching_clause[0]
            watching_clause = watching_clause[1:]
            
            watching_clause = watching_clause[watching_clause!=0]
            # todo: does here too many lits are assigned in the for loop not for else branch?
            for lit in watching_clause:
                if lit in clause2lits[index]:
                    # lit is another watching literal of watching_clause
                    continue
                elif assignments.assigns[abs(lit)-1, 4] == 1 and assignments.value(lit) == False:
                    # lit is a assigned False
                    continue
                else:
                    # lit is not another watching literal of watching_clause
                    # and is non-False literal, so we rewatch it. (case 1)
                    
                    # exchange if clause is not the first element
                    c2l_flag = clause2lits[index] == watching_lit
                    if c2l_flag[1]:
                        clause2lits[index][c2l_flag] = lit
                    else:
                        clause2lits[index][0] = clause2lits[index][1]
                        clause2lits[index][1] = lit

                    # get current index in lit2clauses
                    l2c_flag = lit2clauses[watching_lit_idx][index]
                    cur_row = lit2clauses[watching_lit_idx]
                    cur_row[cur_row > l2c_flag] = cur_row[cur_row > l2c_flag]-1
                    lit2clauses[watching_lit_idx][index] = 0

                    lit2clauses[abs_offset(lit, total_len)][index] = lit2clauses[abs_offset(lit, total_len)][-1]
                    lit2clauses[abs_offset(lit, total_len)][-1] += 1
                    
                    break
            else:
                # we cannot find another literal to rewatch (case 2,3,4)
                watching_lits = clause2lits[index]
                if len(watching_lits) == 1:
                    # watching_clause is unit clause, and the only literal
                    # is assigned False, thus indicates a conflict
                    return ('conflict', watching_clause)
               	
                # the other watching literal
                other = watching_lits[0] if watching_lits[1] == watching_lit else watching_lits[1]
                
                if assignments.assigns[abs(other)-1, 4] != 1:
                    # the other watching literal is unassigned. (case 3)
                    assignments.assign(other, other>0, index)
                    to_propagate.insert(0, other)
                elif assignments.value(other) == True:
                    # the other watching literal is assigned True. (case 2)
                    continue
                else:
                    # the other watching literal is assigned False. (case 4)
                    return ('conflict', watching_clause)

    return ('unresolved', None)


def resolve(a, b, x: int, formula: Formula):
    """
    Performs the resolution operation between two clauses on a pivot literal x.
    
    Return: The resulting clause after resolution.
    """
    items = []
    for i in a:
        items.append(i.item())
    for i in b:
        items.append(i.item())

    res = [i for i in items if not i in [x, -x, 0]]
    
    y = []
    temp = []
    for i in res:
        if i not in temp:
            temp.append(i)
            y.append(resolve_lit(i, formula.hash_val[abs_offset(i, formula.max_len)].item()))
    y = list(set(y))
    result = [i.val for i in y]
   
    result = torch.tensor(list(result))
    return result



def conflict_analysis(clause: torch.tensor, assignments: Assignments, formula: Formula):
    """
    Analyzes a conflict to produce a new learnt clause and determines the backtrack level.
    
    Return: A tuple with the backtrack level and the learnt clause.
    """
    if assignments.dl == 0:
        return (-1, None)
    # literals with current decision level
    clause = clause[1:]
    literals = torch.tensor([literal for literal in clause if assignments.assigns[abs(literal)-1][2] == assignments.dl])
    
    while len(literals) != 1:
        mask = assignments.assigns[abs(literals)-1, 1] != -1
        literals = literals[mask]

        for i in literals:
            if assignments.assigns[abs(i)-1, 1] != -1:
                literal = i
                break
        
        antecedent = assignments.assigns[abs(literal)-1][1]
        
        clause = resolve(clause, formula.clauses[antecedent][1:], abs(literal).item(), formula)
        literals = torch.tensor([literal for literal in clause if assignments.assigns[abs(literal)-1, 2] == assignments.dl])
    # out of the loop, `clause` is now the new learnt clause
    # compute the backtrack level b (second largest decision level)
    decision_levels = sorted(set(assignments.assigns[torch.abs(clause)-1, 2].tolist()))
    if len(decision_levels) <= 1:
        return 0, clause
    else:
        return decision_levels[-2], clause


def parse_dimacs_cnf(content: str, args):
    """
    Parses a CNF formula in DIMACS format and converts it into a Formula object.
    
    Return: A Formula object representing the parsed CNF formula.
    """

    max_len = 0
    lines = 0
    for line in content.splitlines():
        tokens = line.split()
        if tokens and (tokens[0].isdigit() or tokens[0][0] == '-'):
            lines += 1
            if max_len < len(tokens):
                max_len = len(tokens)
    clauses = [[]]

    clauses = [Clause([])]
    for line in content.splitlines():
        tokens = line.split()
        if len(tokens) != 0 and tokens[0] not in ("p", "c"):
            for tok in tokens:
                try:
                    lit = int(tok)
                    if lit == 0:
                        clauses.append(Clause([]))
                    else:
                        var = abs(lit)
                        neg = lit < 0
                        clauses[-1].literals.append(Literal(var, neg))
                except:
                    pass
                    
    if len(clauses[-1]) == 0:
        clauses.pop()
    return Formula(clauses, max_len, args)
