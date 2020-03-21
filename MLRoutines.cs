using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math.Optimization;
using Accord.IO;
using Accord.Statistics.Analysis; // Needed for confusion matrix
using Accord.Math.Optimization.Losses; // needed for use of Zero One Loss routine
using Accord.MachineLearning; // Needed for cross validation
using System;
using Accord.MachineLearning.VectorMachines.Learning;

namespace LearningRountines
{
    class MLAlgorithms
    {

        static public int [] IterativeLeastSquares(double[][] input1, int[] output1, string fName)
        {
            
            double[] labels = System.Array.ConvertAll<int, double>(output1, x => x);
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                // Gets or sets the tolerance value used to determine whether the algorithm has converged.
                Tolerance = 1e-4,  // Let's set some convergence parameters
                MaxIterations = 10,
                //MaxIterations = 100,  // maximum number of iterations to perform
                Regularization = 0
            };

            // Now, we can use the learner to finally estimate our model:
            LogisticRegression regression = learner.Learn(input1, output1);
            double [] coefficients = learner.Solution;

            double[] scores = regression.Probability(input1);
            
            regression.Save(fName.Replace(".csv", ".IRLS.save"), compression: SerializerCompression.None);

            // Finally, if we would like to arrive at a conclusion regarding
            // each sample, we can use the Decide method, which will transform
            // the probabilities (from 0 to 1) into actual true/false values:

            return Funcs.Utility.BoolToInt(regression.Decide(input1));
            
            // mean(double(p == y)) * 100);
        }

        static public int[] MultiNomialLogisticRegressionBFGS (double [][] input, int [] labels, string fName)
        {
            /* The L-BFGS algorithm is a member of the broad family of quasi-Newton optimization methods.
             * L-BFGS stands for 'Limited memory BFGS'. Indeed, L-BFGS uses a limited memory variation of
             * the Broyden–Fletcher–Goldfarb–Shanno (BFGS) update to approximate the inverse Hessian matrix
             * (denoted by Hk). Unlike the original BFGS method which stores a dense approximation, L-BFGS
             * stores only a few vectors that represent the approximation implicitly. Due to its moderate
             * memory requirement, L-BFGS method is particularly well suited for optimization problems with
             * a large number of variables. 
             */
             
            // Create a lbfgs model
            var mlbfgs = new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>();

            // Estimate using the data against a logistic regression
            MultinomialLogisticRegression mlr = mlbfgs.Learn(input, labels);

            // 
            // Create a cross validation model derived from the training set to measure the performance of this
            // predictive model and estimate how well we expect the model will generalize. The algorithm executes
            // multiple rounds of cross validation on different partitions and averages the results. 
            //
            int folds = 4; // could play around with this later
            var cv = CrossValidation.Create(k: folds, learner: (p) => new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>(), 
                loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),
                x: input, y: labels);
            var result = cv.Learn(input, labels);
            
            System.Console.WriteLine(   "Cross Validation Mean {0}, Training Error {1}", result.Validation.Mean, result.Training.Variance);

            GeneralConfusionMatrix gcm = result.ToConfusionMatrix(input, labels);
            System.Console.WriteLine("  General confusion matrix accuracy = {0}%\n", gcm.Accuracy * 100);

            // Compute the model predictions and return the values
            int[] answers = mlr.Decide(input);
            
            // And also the probability of each of the answers
            double[][] probabilities = mlr.Probabilities(input);

            // Now we can check how good our model is at predicting
            double error = new Accord.Math.Optimization.Losses.ZeroOneLoss(labels).Loss(answers);
            mlr.Save(fName, compression: SerializerCompression.None);

            return answers;
        }
        static public  int[] ProbabilisticCoordinateDescent( double[][] input1, int[] labels, string SaveFile)
        {
            // http://accord-framework.net/docs/html/T_Accord_MachineLearning_VectorMachines_Learning_ProbabilisticCoordinateDescent.htm
            /* This class implements a SupportVectorMachine learning algorithm specifically crafted for
             * probabilistic linear machines only. It provides a L1- regularized coordinate descent learning
             * algorithm for optimizing the learning problem. The code has been based on liblinear's method
             * solve_l1r_lr method, whose original description is provided below.
             * 
             * Liblinear's solver -s 6: L1R_LR. A coordinate descent algorithm for L1-regularized logistic
             * regression (probabilistic svm) problems.
             */

            int folds = 5;
            Accord.Math.Random.Generator.Seed = 0;
            var cv = CrossValidation.Create(

                    k: folds, // We will be using 10-fold cross validation

                    // First we define the learning algorithm:
                    learner: (p) => new ProbabilisticCoordinateDescent(),

                    // Now we have to specify how the n.b. performance should be measured:
                    loss: (actual, expected, p) => new ZeroOneLoss(expected).Loss(actual),

                    // This function can be used to perform any special
                    // operations before the actual learning is done, but
                    // here we will just leave it as simple as it can be:
                    fit: (teach, x, y, w) => teach.Learn(x, y, w),

                    // Finally, we have to pass the input and output data
                    // that will be used in cross-validation. 
                    x: input1, y: labels
                    );
            
            var cvresult = cv.Learn(input1, labels);
            Console.WriteLine("  Generating a cross validation for the dataset");
            Console.WriteLine("  Cross Validation samples {0}", cvresult.NumberOfSamples);
            Console.WriteLine("  Cross Validation features {0}", cvresult.NumberOfInputs);
            Console.WriteLine("  Cross Validation Training Mean {0}", cvresult.Training.Mean);
            GeneralConfusionMatrix gcm = cvresult.ToConfusionMatrix(input1, labels);
            Console.WriteLine("  Confusion Matrix Accuracy {0}", gcm.Accuracy);

            
            var teacher = new ProbabilisticCoordinateDescent()
            {
                Tolerance = 1e-10,
                Complexity = 1e+10,
                // learn a hard-margin model
                /* Complexity (cost) parameter C. Increasing the value of C forces the creation of a more
                 * accurate model that may not generalize well. If this value is not set and UseComplexityHeuristic
                 * is set to true, the framework will automatically guess a value for C. If this value is manually
                 * set to something else, then UseComplexityHeuristic will be automatically disabled and the given
                 * value will be used instead.
                 */
            };
            var svm = teacher.Learn(input1, labels);
            var svmregression = (LogisticRegression)svm;
            ConfusionMatrix cm = ConfusionMatrix.Estimate(svm, input1, labels);
            // accuracy, TP, FP, FN, TN and FScore Diagonal

            Console.WriteLine(" Results of Training run");
            Console.WriteLine(" Accuracy:{0}, False Positives: {1}, False Negatives: {2}, Fscore: {3}\n", cm.Accuracy,
                cm.FalsePositives, cm.FalseNegatives, cm.FScore);

            // Write the model out to a save file
            string modelsavefilename = SaveFile.Replace(".csv", ".PCD.save");

            svmregression.Save(modelsavefilename, compression: SerializerCompression.None);

            bool[] answers = svmregression.Decide(input1);
            return Funcs.Utility.BoolToInt(answers);
            
        }

        static public int [] MultiNomialLogRegressionLowerBoundNewtonRaphson (double [][] input1, int[] labels, string SaveFile)
        {
            // http://accord-framework.net/docs/html/T_Accord_Statistics_Models_Regression_MultinomialLogisticRegression.htm
            // Create a estimation algorithm to estimate the regression
            LowerBoundNewtonRaphson lbnr = new LowerBoundNewtonRaphson()
            {
                MaxIterations = 100,
                Tolerance = 1e-6
            };

            // Now, we will iteratively estimate our model:
            MultinomialLogisticRegression mlr = lbnr.Learn(input1, labels);

            // We can compute the model answers
            int[] answers = mlr.Decide(input1);
            string modelsavefile = SaveFile.Replace(".csv", ".MLR.save");

            mlr.Save(modelsavefile, compression: SerializerCompression.None);

            return answers;
        }
    }
}