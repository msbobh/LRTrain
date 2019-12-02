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
using Accord.MachineLearning.VectorMachines.Learning;


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
            // ********************* new stuff to try out *********************

            var teacher = new Accord.MachineLearning.VectorMachines.Learning.ProbabilisticCoordinateDescent()
            {
                Tolerance = 1e-10,
                Complexity = 1e+10, // learn a hard-margin model
            };
            var svm = teacher.Learn(input1, output1);
            var svmregression = (LogisticRegression)svm;
                   
            // Write the model out to a save file
            string modelsavefilename = trainingfile.Replace(".csv", ".PCD.save");
            
            svmregression.Save(modelsavefilename, compression: SerializerCompression.None);

            // Use the Decide method, to  transform
            // the probabilities (from 0 to 1) into actual true/false values:
            bool[] predicted = svmregression.Decide(input1);
            

            int[] svmpredicts = Utility.BoolToInt(predicted); // Decide returns boolean need to conver to ints to compare w/ input labels
            double subtotal = 0;
            int index = 0;
            foreach (var result in svmpredicts)
            {
                if (result == output1[index])
                {
                    subtotal = subtotal + 1;
                }
                index++;
            }
            double svmaccuracy = subtotal / svmpredicts.Count();
            Console.WriteLine("SVM Accuracy:{0}", Math.Round(svmaccuracy * 100, 2));
            // Compute the classification error as in SVM example
            double error = new Accord.Math.Optimization.Losses.ZeroOneLoss(output1).Loss(predicted);
            Console.WriteLine("Zero One Loss:{0}", Math.Round(error, 2));
            /*
             Console.WriteLine("Predicted accuracy:{0}", Math.Round(accuracy * 100, 2));
             */
             /*
              * Run the same data through a BFGS optimizaiton routine and save the model
              * 
              */

            int[] BFGSResults = LearningRountines.MLAlgorithms.ConjugateGradientDescentBFGS(input1, output1, trainingfile.Replace(".csv", ".BFGS.save"));
            index = 0;
            subtotal = 0;
            foreach (var result in BFGSResults)
            {
                if (result == output1[index])
                {
                    subtotal = subtotal + 1;
                }
                index++;
            }
            double BFGSAccuracy = subtotal / BFGSResults.Count();
            Console.WriteLine("BFGS Accuracy => {0}", Math.Round(BFGSAccuracy * 100, 2));


        }
    }
}
