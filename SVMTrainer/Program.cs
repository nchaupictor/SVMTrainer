using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Math;
using Accord.Statistics.Kernels;
using Accord.IO;
using Accord.Controls;
using Accord.Statistics.Analysis;
using System.IO;

namespace SVMTrainer
{
    class Program
    {
        static void Main()
        {
            //Uncomment this entire block for training 
            Console.Write("SVM Trainer");
            //var Root = "C:\\Users\\Pictor17\\Python\\Calibration Slide Wet Check\\New Brighter Settings Izone\\";
            var Root = "Z:\\Protocols&Analysis2008\\Nixon\\66E55 Wet Checker Revisited\\Calibration Slide Wet Check\\New Brighter Settings Izone\\New SVM";
            string FilePath = Root + "\\dataG7.csv";
            System.IO.TextReader reader = new StreamReader(FilePath);
            // Open dataset 
            CsvReader datR = new CsvReader(reader, false);
            double[][] mat = datR.ToTable().ToArray();
            mat = mat.Transpose();
        
            FilePath = Root + "\\labelsG7.csv";
            System.IO.TextReader readerL = new StreamReader(FilePath);
            // Open labels
            CsvReader labR = new CsvReader(readerL, false);
            System.Data.DataTable temp = labR.ToTable();
            double[] dlab = temp.Columns[0].ToArray();
            int[] labels = new int[dlab.Length];
            /* ------------------------------------------------------------------------------------------------------------------*/
            //Convert values of 0 in csv to -1(0 = dry)
            for (int i = 0; i < dlab.Length; i++)
            {
                if (dlab[i] == 0)
                {
                    dlab[i] = -1;
                }
                labels[i] = (int)dlab[i];
            }
            /* ------------------------------------------------------------------------------------------------------------------*/
            //Sort data variables and split into test / train set
            var nRows = mat.Length;
            var nCols = mat[0].Length;
            var nRowsTest = Convert.ToInt32(0.05 * nRows); //95% Train / 5% Test
            var nRowsTrain = nRows - nRowsTest;

            double[][] trainDat = new double[nRowsTrain][];
            int[] y_train = new int[nRowsTrain];
            for (int k = 0; k < nRowsTrain; k++)
            {
                trainDat[k] = new double[nCols];
                Array.Copy(mat[k], trainDat[k], nCols);
                y_train[k] = (int)dlab[k];
            }

            double[][] testDat = new double[nRowsTest][];
            int[] y_test = new int[nRowsTest];
            for (int k = 0; k < nRowsTest; k++)
            {
                testDat[k] = new double[nCols];
                Array.Copy(mat[nRows - nRowsTest + k], testDat[k], nCols);
                y_test[k] = (int)dlab[nRows - nRowsTest + k];
            }
            Console.Write("\nDataset Uploaded");
            double[] pred = new double[nRowsTest];
            double[] predksvm = new double[nRowsTest];


            //Create grid search
            //Accord.MachineLearning.GridSearchRange[] ranges =
            //{
            //    new Accord.MachineLearning.GridSearchRange("complexity" , new double[] {0.000001,0.00001,0.0001, 0.001, 0.01,0.1,0.2,0.3,0.4}),  
            //};

            //Console.Write("\nPerforming Grid Search on SVM");
            //var gridsearch = new Accord.MachineLearning.GridSearch<SupportVectorMachine>(ranges);
            //gridsearch.Fitting = delegate(Accord.MachineLearning.GridSearchParameterCollection parameters, out double error)
            //{
            //    double complexity = parameters["complexity"].Value;
            //    SupportVectorMachine svm = new SupportVectorMachine(mat.Columns());
            //    SequentialMinimalOptimization smo = new SequentialMinimalOptimization(svm, trainDat, y_train);
            //    smo.Complexity = complexity;
            //    error = smo.Run();
            //    return svm;
            //};

            //Console.Write("\nTraining Optimised SVM...");
            //Accord.MachineLearning.GridSearchParameterCollection bestParameters; double minError;
            //SupportVectorMachine bsvm = gridsearch.Compute(out bestParameters, out minError);
            Console.Write("\nTraining ...");
            var crossvalidation = new Accord.MachineLearning.CrossValidation(size: mat.Length, folds: 10);
            crossvalidation.Fitting = delegate (int k, int[] indicesTrain, int[] indicesValidation)
            {
                var trainingInputs = mat.Submatrix(indicesTrain);
                var trainingOutputs = labels.Submatrix(indicesTrain);
                var validationInputs = mat.Submatrix(indicesValidation);
                var validationOutputs = labels.Submatrix(indicesValidation);

                var svm = new SupportVectorMachine(mat.Columns());

                var smo = new SequentialMinimalOptimization(svm, trainingInputs, trainingOutputs);
                smo.Complexity = 0.000001;
                double error = smo.Run();
                double validationError = smo.ComputeError(validationInputs, validationOutputs);
                return new Accord.MachineLearning.CrossValidationValues(svm, error, validationError);
            };

            var result = crossvalidation.Compute();
            double trainingErrors = result.Training.Mean;
            double validationErrors = result.Validation.Mean;
            Console.Write("\nCross Validation Complete");
            var minError = result.Models.Select(y => y.ValidationValue).Min();
            var bestModel = result.Models.Where(x => x.ValidationValue == minError).FirstOrDefault();

            SupportVectorMachine bsvm = (SupportVectorMachine)bestModel.Model;
            

            /* ------------------------------------------------------------------------------------------------------------------*/
            //Initialise and run SVM

            //Accord.MachineLearning.GridSearchRange[] rangesk =
            //{
                //new Accord.MachineLearning.GridSearchRange("complexity" , new double[] {0.000001,0.00001,0.0001, 0.001, 0.01,0.1,1}),
                //new Accord.MachineLearning.GridSearchRange("gamma", new double[] {0.0001,0.001,0.01, 0.1}),
            //};
            //Console.Write("\nPerforming Grid Search on kSVM");
            //var gridsearchk = new Accord.MachineLearning.GridSearch<KernelSupportVectorMachine>(rangesk);
            //gridsearchk.Fitting = delegate (Accord.MachineLearning.GridSearchParameterCollection parametersk, out double errork)
            //{
                //double complexity = parametersk["complexity"].Value;
                //double gamma = parametersk["gamma"].Value;
                //Gaussian gauss = new Gaussian();
                //gauss.Gamma = gamma;
                //KernelSupportVectorMachine ksvm = new KernelSupportVectorMachine(gauss, mat.Columns());
                //SequentialMinimalOptimization ksmo = new SequentialMinimalOptimization(ksvm, trainDat, y_train);
                //ksmo.Complexity = complexity;
                //errork = ksmo.Run();
                //return ksvm;
            //};

            //Console.Write("\nTraining Optimised kSVM...");
            //Accord.MachineLearning.GridSearchParameterCollection bestParametersk; double minErrork;
            //KernelSupportVectorMachine bksvm = gridsearchk.Compute(out bestParametersk, out minErrork);

            //double error = smo.Run();
            //double errork = ksmo.Run();
            Console.Write("\nTraining Complete");
            //Console.Write("\nBest C: %0.2f", bestParameters[0].Value);
            /* ------------------------------------------------------------------------------------------------------------------*/
            //Save SVM Model
            System.Runtime.Serialization.Formatters.Binary.BinaryFormatter formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            FileStream Savestream = new FileStream("svm.bin", FileMode.Create);
            formatter.Serialize(Savestream, bsvm);
            Savestream.Close();
            Console.Write("\nSaved SVM Model");

            //System.Runtime.Serialization.Formatters.Binary.BinaryFormatter formatterk = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            //FileStream Savestreamk = new FileStream("ksvm.bin", FileMode.Create);
            //formatterk.Serialize(Savestreamk, bksvm);
            //Savestreamk.Close();
            //Console.Write("\nSaved kSVM Model");

            //string tPath = String.Format(@"{0}\Pictorial\Files\svm.bin", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            //string tPath = "svm.bin"; //Local project location of SVM model 
            //Predict values
            for (int i = 0; i < nRowsTest; i++) //Uncomment for training                     
            {
                pred[i] = bsvm.Compute(testDat[i]); //Uncomment for training
                //predksvm[i] = bksvm.Compute(testDat[i]);
            }
            /* ------------------------------------------------------------------------------------------------------------------*/
            // Use confusion matrix to compute some performance metrics
            int[] predict = new int[pred.Length];
            int[] predictk = new int[predksvm.Length];
            for (int p = 0; p < pred.Length; p++)
            {
                predict[p] = Math.Sign(pred[p]);
                //predictk[p] = Math.Sign(predksvm[p]);
            }
            ConfusionMatrix confusionMatrix = new ConfusionMatrix(predict, y_test, -1, 1); //Predicted, Expected, Positive, Negative Uncomment for training     
            //ConfusionMatrix confusionMatrixk = new ConfusionMatrix(predictk, y_test, -1, 1);
            Console.Write("Results:\n");
            Console.Write("\nSamples: " + confusionMatrix.Samples);
            Console.Write("\nSensitivity: " + confusionMatrix.Sensitivity);
            Console.Write("\nSpecificity: " + confusionMatrix.Specificity);
            Console.Write("\nAccuracy: " + confusionMatrix.Accuracy);
            Console.ReadKey();
        }
    }
}
