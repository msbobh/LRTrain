﻿using System;
using Accord.Statistics.Models.Regression;
using Accord.Math;
using Accord.IO;
using Funcs;
using resources;
using LearningRountines;



namespace AccordLogisticRegression
{
    class Program
    {
        
        static void Main(string[] args)
        {
            /* 
             * some declartions
             */
            string trainingfile = null;
            string labelfile = null;
            const int minargs = 2;
            const int maxargs = 3;
            
            int numArgs = Utility.parseCommandLine(args, maxargs, minargs);
            if (numArgs == 0)
            {
                Console.WriteLine(strings.usage);
                System.Environment.Exit(1);
            }
                        
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
                       
                            
            Console.WriteLine(" Logistic Regression (Accord.net) Training Utility Starting...\n");
            //Console.WriteLine("Learning 3 differernt Models: Profbablistic Coordinate Descent, Iterative Reweighted Least Squares, Conjugate Gradient Descent (BFGS)\n");

            // 
            // Read in the training file an convert to a Matrix
            //
            CsvReader training_samples = new CsvReader(trainingfile, false);
            double[,] MatrixIn = training_samples.ToMatrix<double>();
            
            int rows = MatrixIn.Rows();
            int cols = MatrixIn.Columns();
        
            // 
            // Read in the label file an convert to a Matrix
            //
            CsvReader labelscsv = new CsvReader(labelfile, false);
            double[,] labels = labelscsv.ToMatrix<double>();
            
            if (rows != labels.Rows()) // number of samples must match
            {
                Console.WriteLine(strings.SampleMisMatch, cols, 4);
                System.Environment.Exit(1);
            }
                       
            // For Accord.net Logistic Regression the input data needs to be in Jagged Arrays         
            double[][] input1 = MatrixIn.ToJagged<double>();
            
            // Labels can either be int (1,0) or bools
            int [] output1 = Utility.convetToJaggedArray(labels);

            // Learn a  Probabilistic Coordinate Descent model
            //

            // commenting this routine out, as it is a linear solver
            /*
            Console.WriteLine("Starting Probabilistic Gradient Descent");
            int[] svmpredicts = MLAlgorithms.ProbabilisticCoordinetDescent (input1, output1, trainingfile);
            double svmaccuracy = Funcs.Utility.CalculateAccuraccy(svmpredicts, output1);
                        
            Console.WriteLine(" Probablistic Coordinate Descent w/SVM Accuracy:{0}", Math.Round(svmaccuracy * 100, 2));
            // Compute the classification error as in SVM example
            double error = new Accord.Math.Optimization.Losses.ZeroOneLoss(output1).Loss(svmpredicts);
            Console.WriteLine("Zero One Loss:{0}\n", Math.Round(error, 2));
            */

            Console.WriteLine("Starting Multinomial Logistic Regression using L-BFGS");
            int[] BFGSPredicts = MLAlgorithms.MultiNomialLogisticRegressionBFGS (input1, output1, trainingfile.Replace(".csv", ".BFGS.save"));
            double BFGSAccuracy = Utility.CalculateAccuraccy (BFGSPredicts, output1);
            Console.WriteLine(" Training Accuracy => {0}%\n", Math.Round(BFGSAccuracy * 100, 2));

            // Commenting this algorithm out, after running for a few hours on a 25 sample resume file it got an out of memeory error
            
            /*
            Console.WriteLine("Starting Iterative Reweighted Least Squares");
            int[] IRLSPredicts = MLAlgorithms.IterativeLeastSquares(input1, output1, trainingfile);
            double IRLSAccuracy = Utility.CalculateAccuraccy (IRLSPredicts, output1);
            Console.WriteLine(" Iterative Least Squares (IRLS)\nAccuracy => {0}\n", Math.Round(IRLSAccuracy * 100, 2));
            */

            // Commenting out this method, it is too long running on the resume data set.

            /* Console.WriteLine("starting Multinomial Log Regression w/ Lowerbound Newton Raphson");
            int[] MNLRPredicts = MLAlgorithms.MultiNomialLogRegressionLowerBoundNewtonRaphson(input1, output1, trainingfile);
            double MNLRAccuracy = Funcs.Utility.CalculateAccuraccy(MNLRPredicts, output1);
            Console.WriteLine("Multinomial Logistic Regression using LB Newton Raphson (MNLR)\nAccuracy => {0}", Math.Round(MNLRAccuracy * 100, 2));
            */
        }
    }
}
