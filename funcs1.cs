using System;
using Accord.Math;
using System.IO;


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
           
    }

}