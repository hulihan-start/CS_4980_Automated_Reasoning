# CS_4980_Automated_Reasoning 
## THCDCL: a CDCL solver with pytorch tensor implementation

This is course project for [CS:4980 Introduction to Automated Reasoning](https://homepage.cs.uiowa.edu/~tinelli/classes/4980/Spring24/) (Spring 24) by Professor [Cesare Tinelli](https://homepage.cs.uiowa.edu/~tinelli/).

Group member: Hua Chai, Zhengyang He and [Lihan Hu](https://hulihan-start.github.io/).

## Requirements

- Python 3.8 or higher
- PyTorch 1.8 or higher
- A GPU with CUDA support (optional)

## Installation

Before running the solver, ensure that you have Python and PyTorch installed on your system. You can download Python from [python.org](https://www.python.org/downloads/) and install PyTorch using pip with GPU support by following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/).

### Installing Python packages

After installing Python, you can install the required Python packages using pip:

```bash
pip install torch
```

If you are on a machine with CUDA (NVIDIA GPU), make sure to install the CUDA version of PyTorch that corresponds to your CUDA version.

## Usage

To use the CDCL SAT Solver, you need to provide a CNF file in DIMACS format. You can run the solver using the following command.

### Basic Command

Navigate to the directory containing the solver's code and run:

```bash
python main.py --cnf_file FILE_PATH
```

Replace `FILE_PATH` with the path to your CNF file.

### GPU Acceleration

Our THCDCL also support reasoning phase on GPU. Our current GPU implementation may not be faster than on CPU, but we believe this implementation will be beneficial for more complex automated reasoning scenario. If you want to try our GPU version, please use this command:

```bash
python main.py --cnf_file FILE_PATH --gpu
```

## Output

The solver will output whether the CNF is SATISFIABLE or UNSATISFIABLE. If the formula is satisfiable, the solver will also print the assignments that satisfy the formula.
