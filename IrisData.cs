using Accord.DataSets;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.MachineLearning.VectorMachines;
using Accord.Statistics.Analysis;
using Accord.Statistics.Kernels;
using Accord.Math.Random;
using System;

namespace LRTrain
{
    class IrisData
    {
        public IrisData (in int rando)
        {
            // Generate always same random numbers
           Generator.Seed = rando;

            // Download and load the Iris dataset
            Iris iris = new Iris();
            double[][] inputs = iris.Instances;
            int[] outputs = iris.ClassLabels;           

        }
        
    }
}
