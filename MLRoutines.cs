using Accord.MachineLearning;
using Accord.Statistics;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math.Optimization;
using Accord.IO;


namespace LearningRountines
{
    class MLAlgorithms
    {

        static public bool [] IterativeLeastSquares(double[][] input1, double[] output1, string fName)
        {
            
            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                // Gets or sets the tolerance value used to determine whether the algorithm has converged.
                Tolerance = 1e-4,  // Let's set some convergence parameters
                MaxIterations = 100,  // maximum number of iterations to perform
                Regularization = 0
            };

            // Now, we can use the learner to finally estimate our model:
            LogisticRegression regression = learner.Learn(input1, output1);

            double[] scores = regression.Probability(input1);
            
            regression.Save(fName.Replace(".csv", ".IRLS.save"), compression: SerializerCompression.None);

            // Finally, if we would like to arrive at a conclusion regarding
            // each sample, we can use the Decide method, which will transform
            // the probabilities (from 0 to 1) into actual true/false values:

            return regression.Decide(input1);
            // mean(double(p == y)) * 100);
        }

        static public int[] ConjugateGradientDescentBFGS (double [][] input, int [] labels, string fName)
        {
            // Create a Conjugate Gradient algorithm to estimate the regression
            var mlbfgs = new MultinomialLogisticLearning<BroydenFletcherGoldfarbShanno>();
            
            // Now, we can estimate our model using BFGS
            MultinomialLogisticRegression mlr = mlbfgs.Learn(input, labels);

            // We can compute the model answers
            int[] answers = mlr.Decide(input);
            
            // And also the probability of each of the answers
            double[][] probabilities = mlr.Probabilities(input);

            // Now we can check how good our model is at predicting
            double error = new Accord.Math.Optimization.Losses.ZeroOneLoss(labels).Loss(answers);
            mlr.Save(fName, compression: SerializerCompression.None);

            return answers;
        }
        
    }
}