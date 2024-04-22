from utils import *
import time


if __name__ == '__main__':
    # you might comment it to get inconsistent execution time
    random.seed(5201314)

    if len(sys.argv) != 2:
        print('Provide one DIMACS cnf filename as argument.')
        sys.exit(1)
        
    dimacs_cnf = open(sys.argv[1]).read()
    start = time.time()
    formula = parse_dimacs_cnf(dimacs_cnf)

    # print(formula, formula.clauses, formula.variables())
    
    # time1 = time.time()
    result = cdcl_solve(formula)
    # print(result)
    # time2 = time.time()
    # if result:
    #     assert result.satisfy(formula)
    #     print('Formula is SAT with assignments:')
    #     assignments = {var: assignment.value for var, assignment in result.items()}
    #     pprint(assignments)
    # else:
    #     print('Formula is UNSAT.')
    # print('time cost: {}, {}, {}'.format(time.time() - time2, time2-time1, time1-start))