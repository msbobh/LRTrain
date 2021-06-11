Logistic Regression Training Routine (Accord.net)
======
**LRTrain** Logistic Regression Training routine. Required Accord.net: Math, IO and Statistical Routines, saves a 
trained regression model to disk.

LRTrain:
Loads a training file specified from the CL
Loads a label file specified from the CL 

The files are required to be in csv format, conversion to the appropriate format is done internal to the program
It then trains a Logistic Regressions from the supplied data, using a library from Accord.net.  The following algorithms are invoked:
Multinomial Logistic Regression BFGS
Iterative Least Squares(Note this algorithm is very slow on data sets with large numbers of features)
Multinomial Logistic Regression Lower Bound Newton Raphson.

The models are saved in the following format <trainfilename>.modeltype.save where modeltype = BFGS | PCD | IRLS)
  
## Accord Routines

The L-BFGS algorithm is a member of the broad family of quasi-Newton optimization methods. L-BFGS stands for 'Limited memory BFGS'. Indeed, L-BFGS uses a limited memory variation of the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update to approximate the inverse Hessian matrix (denoted by Hk). Unlike the original BFGS method which stores a dense approximation, L-BFGS stores only a few vectors that represent the approximation implicitly. Due to its moderate memory requirement, L-BFGS method is particularly well suited for optimization problems with a large number of variables. 
               

 http://accord-framework.net/docs/html/T_Accord_MachineLearning_VectorMachines_Learning_ProbabilisticCoordinateDescent.htm
 This class implements a SupportVectorMachine learning algorithm specifically crafted for probabilistic linear machines only. It provides a L1- regularized coordinate descent learning algorithm for optimizing the learning problem. The code has been based on liblinear's method solve_l1r_lr method, whose original description is provided below. Liblinear's solver -s 6: L1R_LR. A coordinate descent algorithm for L1-regularized logisticregression (probabilistic svm) problems.
               
## Contributors
Bob Hildreth

### Third party libraries
* Uses Accord.Net 
* http://accord-framework.net/

## Version 
* Version 0.1

## Contact
#### Bob/HAL/R
* Homepage: 
* e-mail: Bobh@thehildreths.com


