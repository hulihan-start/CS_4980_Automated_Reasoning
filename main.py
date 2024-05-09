from utils import  parse_dimacs_cnf, cdcl_solve
import argparse
import random

parser = argparse.ArgumentParser(description="CDCL arguments")
parser.add_argument('--gpu', action='store_true', help='add our GPU implementation')
parser.add_argument('--cnf_file', type=str, help='file path')
args = parser.parse_args()

if __name__ == '__main__':
    # you might comment it to get inconsistent execution time
    random.seed(5201314)
        
    with open(args.cnf_file, 'r') as file:
        dimacs_cnf = file.read()

    formula = parse_dimacs_cnf(dimacs_cnf, args)
    result = cdcl_solve(formula, args)
    
    if result is not None and result.numel() != 0:
        if not result[result[:, 4] == 1] is None:
            print('s SATISFIABLE')
            assignments = [f'{var.item()}' if result[i][0]==1 else f'-{var.item()}' for i, var in enumerate(formula.variables())]
            print(' '.join(assignments))
    else:
        print('s UNSATISFIABLE')