using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math;
using Accord.IO;
using Funcs;


namespace AccordLogisticRegression
{
    class Program
    {
        
        static void Main(string[] args)
        {
            Console.WriteLine(" Running (Accord.net) logistic Regression ");
            string trainingfile = args[0];
            string labelfile = args[1];
            
            CsvReader training_samples = new CsvReader(trainingfile, false);
            //int cols = training_samples.Columns();
            double[,] MatrixIn = training_samples.ToMatrix<double>();
            int rows = MatrixIn.Rows();
            int cols = MatrixIn.Columns();

            CsvReader labelscsv = new CsvReader(labelfile, false);
            double[,] labels = labelscsv.ToMatrix<double>();
                       

            // For Accord.net Logistic Regression the input data needs to be in Jagged Arrays         
            double [][] input1 = Funcs.Utility.convertToJaggedArray(MatrixIn);
            // Labels can either be int (1,0) or bools
            int [] output1 = Utility.convetToJaggedArray(labels);

               

            var learner = new IterativeReweightedLeastSquares<LogisticRegression>()
            {
                                   // Gets or sets the tolerance value used to determine whether the algorithm has converged.
                Tolerance = 1e-4,  // Let's set some convergence parameters
                MaxIterations = 100,  // maximum number of iterations to perform
                Regularization = 0
            };

            // Now, we can use the learner to finally estimate our model:
            LogisticRegression regression = learner.Learn(input1, output1);

            
            // Write the model out to a save file

            double[] scores = regression.Probability(input1);
            string modelsavefilename = "RegressionModel.save";
            regression.Save(modelsavefilename, compression: SerializerCompression.None);

            // Finally, if we would like to arrive at a conclusion regarding
            // each sample, we can use the Decide method, which will transform
            // the probabilities (from 0 to 1) into actual true/false values:

            bool [] actual = regression.Decide(input1);
            // mean(double(p == y)) * 100);

            double[,] learnedParams = Utility.converRowtoColumn( regression.Weights);

            string filename = "LearnedParams.csv";
                                 
            
            using (CsvWriter writer = new CsvWriter(filename))
            {
                writer.Write(learnedParams);
            }
            
            int[] predictions = Utility.BoolToInt(actual);           
            double subtotal = 0;
            int index = 0;
            foreach (var result in predictions)
            {
                if(result == output1[index])
                {
                    subtotal = subtotal + 1;
                }
                index++;
            }
            double accuracy =  subtotal / predictions.Count();
            Console.WriteLine("Predicted accuracy:{0}", Math.Round(accuracy * 100, 2));
            

                
        }
    }
}
