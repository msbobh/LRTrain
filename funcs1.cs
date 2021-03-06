﻿using System;
using System.Linq;
using Accord.Math;
using System.IO;
using resources;


namespace Funcs 
{
    class Utility
    {
        static public double[][] convertToJaggedArray(double[,] multiArray)
        {
            int numOfColumns = multiArray.Columns();
            int numOfRows = multiArray.Rows();

            double[][] jaggedArray = new double[numOfRows][];

            for (int r = 0; r < numOfRows; r++)
            {
                jaggedArray[r] = new double[numOfColumns];
                for (int c = 0; c < numOfColumns; c++)
                {
                    jaggedArray[r][c] = multiArray[r, c];
                }
            }

            return jaggedArray;
        }

        static public int [] convetToJaggedArray(double [,] inputmatrix)
        {
            int numOfRows = inputmatrix.Rows();
            int[] converted = new int[numOfRows];
            for (int r = 0; r < numOfRows; r++)
            {
                converted[r] = (int)inputmatrix[r, 0];
            }
            return converted;
        }

        static public int[] BoolToInt (bool[] input)
        {
            int[] result = new int[input.Rows()];
            int count = 0;
            foreach (var value in input)
            {
                result [count] = Convert.ToInt32(value);
                count++;
            }

            return result;
        }

        static public double[,] converRowtoColumn(double[] input)
        {
            double[,] done = new double[input.Length, 1];
            for (int i = 0; i < input.Length; i++)
            {
                done[i, 0] = input[i];

            }
            return done;
        }

        static public int parseCommandLine(string[] cLine, int maxArgs, int minArgs)
        {
            int numArgs = cLine.Length;
            if (numArgs > maxArgs | numArgs < minArgs)
            {
                return 0;
            }
            switch (numArgs)
            {
                case 1:
                    return 1;

                case 2:
                    return 2;
                case 3:
                    return 3;
                case 4:
                    return 4;

                default:
                    return 0;

            }

        }

        static public bool checkFile(string fname)
        {
            try
            {
                FileStream fs = File.Open(fname, FileMode.Open, FileAccess.Write, FileShare.None);
                fs.Close();
                return true;
            }
            catch (Exception e)
            {
                return false;
            }
        }

        static public double CalculateAccuraccy(int[] labels, int[] Predictions)
        {
            
            int index = 0;
            double subtotal = 0;
            foreach (var result in Predictions)
            {
                if (result == labels[index])
                {
                    subtotal = subtotal + 1;
                }
                index++;
            }

            double Accuracy = subtotal / Predictions.Count();
            return Accuracy;
        }

        static public void Printcolor(int title, ConsoleColor color)
        {
            ConsoleColor originalColor = Console.ForegroundColor;
            Console.ForegroundColor = color;
            Console.WriteLine(title);
            Console.ForegroundColor = originalColor;
        }
        static public void Printcolor (double title, ConsoleColor color)
        {
            ConsoleColor originalColor = Console.ForegroundColor; 
            Console.ForegroundColor = color;
            Console.WriteLine(title);
            Console.ForegroundColor = originalColor;
        }

        static public void OutPutStats( in int Numsamples,in  int Numinputs, in double TrMean, in double GCMAccuracy,
            in int CMFalsePos, in int CMFalseNeg,in double CMFscore)
        {
            Console.WriteLine("  Generating a cross validation for the dataset");
            Console.Write(strings.CrossVSamples);
            Funcs.Utility.Printcolor(Numsamples, ConsoleColor.Yellow);
            Console.Write(strings.CrossValFeatures);
            Funcs.Utility.Printcolor(Numinputs, ConsoleColor.Yellow);
            Console.Write(strings.CrossTrainMean);
            Funcs.Utility.Printcolor( TrMean, ConsoleColor.Yellow);
            Console.Write(strings.ConfusionAcc);
            Funcs.Utility.Printcolor(Math.Round(GCMAccuracy * 100, 2), ConsoleColor.Red);
            Console.WriteLine(strings.TrResults);
            Console.Write(strings.FalsePos);
            Funcs.Utility.Printcolor(CMFalsePos, ConsoleColor.Red);
            Console.Write(strings.FalseNeg);
            Funcs.Utility.Printcolor(CMFalseNeg, ConsoleColor.Red);
            Console.Write(strings.Fscore);
            Funcs.Utility.Printcolor(Math.Round(CMFscore, 2), ConsoleColor.Red);
            Console.WriteLine();
        }

    }

}