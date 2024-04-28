# CS_4980_Automated_Reasoning 
## THCDCL: a CDCL solver with pytorch tensor implementation

This is course project for [CS:4980 Introduction to Automated Reasoning](https://homepage.cs.uiowa.edu/~tinelli/classes/4980/Spring24/) (Spring 24) by Professor [Cesare Tinelli](https://homepage.cs.uiowa.edu/~tinelli/).

Group member: Hua Chai, Zhengyang He and [Lihan Hu](https://hulihan-start.github.io/).

To run our CDCL implementation, use this:
```bash
python main.py --cnf_file FILE_PATH
```

Our THCDCL also support reasoning phase on GPU. Our current GPU implementation may not be faster than on CPU, but we believe this implementation will be beneficial for more complex automated reasoning scenario. If you want to try our GPU version, please use this command:
```bash
python main.py --cnf_file FILE_PATH --gpu
```