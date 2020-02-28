Logistic Regression Training Routine (Accord.net)
======
**LRTrain** Logistic Regressin Training routine. Required Accord.net: Math, IO and Statistical Routines, saves a 
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

## Usage
```$ git clone https://github.com/msbobh/LRTrain.git
...
```
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


