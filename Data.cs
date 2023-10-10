using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;
using System;
using Accord.DataSets;
using System.IO;

namespace LRTrain
{
    //class CreateMNISTData
    // Modified National Institute of Standards and Technology database of handwritten digits
    /* The MNIST database contains 60,000 training images and 10,000 testing images.Half of the 
     * training set and half of the test set were taken from NIST's training dataset, while the 
     * other half of the training set and the other half of the test set were taken from NIST's 
     * testing dataset.[9] The original creators of the database keep a list of some of the 
     * methods tested on it.[7] In their original paper, they use a support-vector machine to 
     * get an error rate of 0.8% 
            {
            public  CreateMINISTData(in int foo)
                {
                    Console.WriteLine("Downloading dataset");
                    string _downloadedFilename = "MNIST Data";
                    string CurrentDir = @"C:\Temp\";
                    Directory.SetCurrentDirectory(CurrentDir);
                    MNIST.Download("http://yann.lecun.com/exdb/mnist/.", CurrentDir, out _downloadedFilename);
                    var _mnist = new Accord.DataSets.MNIST(CurrentDir);
                    System.Tuple<Accord.Math.Sparse<double> [], double[]> _downloaded;
                    _downloaded = _mnist.Training;            

                 }
            }*/

    class CreateMNISTDataset
    {
        public CreateMNISTDataset(in string URL)
        {
            Console.WriteLine("Downloading dataset from: {0}",URL );
            string _downloadedFilename = "MNIST Data";
            string CurrentDir = System.IO.Directory.GetCurrentDirectory();
            Directory.SetCurrentDirectory(CurrentDir);
            MNIST.Download("http://yann.lecun.com/exdb/mnist/.", CurrentDir, out _downloadedFilename);
            var _mnist = new Accord.DataSets.MNIST(CurrentDir);
            System.Tuple<Accord.Math.Sparse<double>[], double[]> _downloaded;
            _downloaded = _mnist.Training;
        }
    }
}
