from utils import *
import time
import argparse

parser = argparse.ArgumentParser(description="CDCL arguments")

parser.add_argument('--gpu', action='store_true', help='add our GPU implementation')
parser.add_argument('--cnf_file', type=str, help='file path')

args = parser.parse_args()
print(args.gpu)

if __name__ == '__main__':
    # you might comment it to get inconsistent execution time
    random.seed(5201314)
        
    dimacs_cnf = open(args.cnf_file).read()
    start = time.time()
    formula = parse_dimacs_cnf(dimacs_cnf, args)
    
    time1 = time.time()
    result = cdcl_solve(formula, args)
    time2 = time.time()
    if not result is None:
        print(result[result[:, 4] == 1])
        if not result[result[:, 4] == 1] is None:
            # assert result.satisfy(formula)
            print('Formula is SAT with assignments:')
            assignments = {var.item(): True if result[i][0]==1 else False for i, var in enumerate(formula.variables())}
            pprint(assignments)
    else:
        print('Formula is UNSAT.')
    print('time cost: {}, {}, {}'.format(time.time() - time2, time2-time1, time1-start))