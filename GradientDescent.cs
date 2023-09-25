using Accord.Statistics.Models.Regression.Fitting;
using Accord.Statistics.Models.Regression;
using System;
using System.Diagnostics;
using System.Security.Cryptography.X509Certificates;
using Accord.MachineLearning;
using Accord.Math.Optimization.Losses;
using Accord.Statistics.Analysis;
using Accord.Math;
using System.Linq;
using Accord.MachineLearning.Performance;
using Accord.Statistics.Models.Markov;
using ICSharpCode.SharpZipLib.BZip2;
using Accord.MachineLearning.DecisionTrees.Learning;

namespace LRTrain
{
    internal class GradientDescent
    {
        public TimeSpan duration { get { return _timespan;  } }
        public int folds { get; set; }
        public LogisticRegression RegressionObject { get => _regressionObj; }

        
        private static TimeSpan _timespan;
        private static LogisticRegression _regressionObj;

        // Constructor: Requires training data and labels trains using Gradient Descent
        public GradientDescent (in double[][] inputs, in double[] labels)
        {

            // Setup a timer to measure training time
            Stopwatch stopWatch = new Stopwatch();
            
            Console.WriteLine("Create a Logistic Regression Model trained using Gradient Descent");
            stopWatch.Start();

            _regressionObj = new LogisticRegression();
            _regressionObj.NumberOfInputs = inputs.Length;
            Accord.Statistics.Models.Regression.Fitting.LogisticGradientDescent learner;
            
            learner = new LogisticGradientDescent(_regressionObj)
            {
                Stochastic = false,
                LearningRate = 1e-4,
            };
            _regressionObj = learner.Learn(inputs, y: labels);
            stopWatch.Stop();
            _timespan = stopWatch.Elapsed;
            

           

        }
    }
}
