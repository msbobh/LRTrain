using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using Accord.Math;
using Accord.IO;
using Funcs;
using resources;


namespace AccordLogisticRegression
{
    class Program
    {
        
        static void Main(string[] args)
        {
            const int minargs = 2;
            const int maxargs = 3;
            int numArgs = Utility.parseCommandLine(args, maxargs, minargs);
            if (numArgs == 0)
            {
                Console.WriteLine(strings.usage);
                System.Environment.Exit(1);
            }
            string trainingfile = null;
            string labelfile = null;
            
            if (numArgs == 2)
            {
                trainingfile = args[0];
                labelfile = args[1];

            }
             if (numArgs == 3) // no use for third parameter yet!
            {
                Console.WriteLine(strings.usage);
                System.Environment.Exit(1);
            }

            if (!Utility.checkFile(trainingfile))
            {
                Console.WriteLine("Error opening file{0}", trainingfile);
                System.Environment.Exit(1);

            }
            if (!Utility.checkFile(labelfile))
            {
                Console.WriteLine("Error opening file {0}", labelfile);
                System.Environment.Exit(1);
            }
                       
                            
            Console.WriteLine(" Logistic Regression (Accord.net) Training Utility");
                        
            // Read in the training file and convert to a Matrix
            CsvReader training_samples = new CsvReader(trainingfile, false);
            double[,] MatrixIn = training_samples.ToMatrix<double>();
            
            int rows = MatrixIn.Rows();
            int cols = MatrixIn.Columns();
        
            // Read in the label file an convert to a Matrix
            CsvReader labelscsv = new CsvReader(labelfile, false);
            double[,] labels = labelscsv.ToMatrix<double>();
            
            if (rows != labels.Rows())
            {
                Console.WriteLine(strings.SampleMisMatch, cols, 4);
                System.Environment.Exit(1);
            }
                       
            // Apparantly if import direclty to Jagged from the csv Accord does not parse the data correctly.
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
            string modelsavefilename = trainingfile.Replace(".csv", ".save");
            double[] scores = regression.Probability(input1);
            
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
