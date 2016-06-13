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
using Accord.Neuro;
using AForge.Neuro;

namespace SVMTrainer
{
    class Program
    {
        static void Main()
        {
            //Uncomment this entire block for training 
            Console.Write("SVM Trainer");
            var Root = "C:\\Users\\Pictor17\\Python";
            string FilePath = Root + "\\Wetdata3s.csv";
            System.IO.TextReader reader = new StreamReader(FilePath);
            // Open dataset 
            CsvReader datR = new CsvReader(reader, false);
            double[][] mat = datR.ToTable().ToArray();
            mat = mat.Transpose();


            double[][] matPCA = new double[mat.Length][];
            mat.CopyTo(matPCA, 0);

            FilePath = Root + "\\Wetlabels3.csv";
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
            // Perform PCA on dataset to reduce features
            //var pca = new PrincipalComponentAnalysis(matPCA, AnalysisMethod.Center);
            //Console.Write("\nPerforming Mean-centred PCA");
            //pca.Compute();
            //double[][] matPCAtransform = pca.Transform(matPCA, 300);
            System.Runtime.Serialization.Formatters.Binary.BinaryFormatter formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            
            //Transform WellROI from PCA
            //string pPath = String.Format(@"{0}\Pictorial\Files\pca.bin", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            //try
            //{
                //using (FileStream LoadPCAStream = File.Open(pPath, FileMode.Open, FileAccess.Read))
                //{
                    //var pca = (PrincipalComponentAnalysis)formatter.Deserialize(LoadPCAStream);
                    //Take 100 most significant features
                    //matPCAtransform = pca.Transform(mat, (int)300);
               // }
            //}
            //catch (IOException)
            //{
            //}



            /* ------------------------------------------------------------------------------------------------------------------*/
            //Sort data variables and split into test / train set
            var nRows = mat.Length;
            var nCols = mat[0].Length;
            var nRowsTest = Convert.ToInt32(0.05 * nRows); //95% Train / 5% Test
            var nRowsTrain = nRows - nRowsTest;

            double[][] trainDat = new double[nRowsTrain][];
            int[] y_train = new int[nRowsTrain];
            //double[] y_train = new double[nRowsTrain];
            for (int k = 0; k < nRowsTrain; k++)
            {
                trainDat[k] = new double[nCols];
                Array.Copy(mat[k], trainDat[k], nCols);
                y_train[k] = (int)dlab[k];
                //y_train[k] = dlab[k];
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

            double[][] outputs = new double[y_train.Length][];
            for (int k = 0; k < y_train.Length ; k++){
                outputs[k] = new double[] { 0 };
            }

            for (int j = 0; j < y_train.Length; j++)
            {
                outputs[j][0] = y_train[j];
            }


            //int numInputs = 36300;
            //int numClasses = 2;
            //int hidden = 1;
            ////double[][] outputs = Accord.Statistics.Tools.Expand(y_train, numClasses, -1, 1);

            //ActivationNetwork network = new ActivationNetwork(new SigmoidFunction(), numInputs, hidden, 1);
            //Accord.Neuro.Learning.LevenbergMarquardtLearning teacher = new Accord.Neuro.Learning.LevenbergMarquardtLearning(network);
            //for(int i = 0; i < 10; i++)
            //{
            //    double error = teacher.RunEpoch(trainDat, outputs);

            //}
            //Gaussian gauss = new Gaussian();
            //gauss.Gamma = 0.01;
            //Quadratic quad = new Quadratic(1);
            //KernelSupportVectorMachine ksvm = new KernelSupportVectorMachine(quad,mat.Columns());
            //SequentialMinimalOptimization ksmo = new SequentialMinimalOptimization(ksvm, trainDat, y_train);
            //ksmo.Complexity = 0.0001;
            //double error = ksmo.Run();

            //Create grid search
            Accord.MachineLearning.GridSearchRange[] ranges =
            {
                new Accord.MachineLearning.GridSearchRange("complexity" , new double[] {1E-4,0.5E-4,1E-3}),
            };

            Console.Write("\nPerforming Grid Search on SVM");
            var gridsearch = new Accord.MachineLearning.GridSearch<SupportVectorMachine>(ranges);
            gridsearch.Fitting = delegate (Accord.MachineLearning.GridSearchParameterCollection parameters, out double error)
            {
                double complexity = parameters["complexity"].Value;
                SupportVectorMachine svm = new SupportVectorMachine(mat.Columns());
                SequentialMinimalOptimization smo = new SequentialMinimalOptimization(svm, trainDat, y_train);
                //ProbabilisticCoordinateDescent smo = new ProbabilisticCoordinateDescent(svm, trainDat, y_train);
                smo.Complexity = complexity;
                error = smo.Run();
                return svm;
            };

            Console.Write("\nTraining Optimised SVM...");
            Accord.MachineLearning.GridSearchParameterCollection bestParameters; double minError;
            SupportVectorMachine bsvm = gridsearch.Compute(out bestParameters, out minError);

            var crossvalidation = new Accord.MachineLearning.CrossValidation(size: mat.Length, folds: 10);
            crossvalidation.Fitting = delegate (int k, int[] indicesTrain, int[] indicesValidation)
            {
                var trainingInputs = mat.Submatrix(indicesTrain);
                var trainingOutputs = labels.Submatrix(indicesTrain);
                var validationInputs = mat.Submatrix(indicesValidation);
                var validationOutputs = labels.Submatrix(indicesValidation);

                var svm = new SupportVectorMachine(mat.Columns());

                var smo = new SequentialMinimalOptimization(svm, trainingInputs, trainingOutputs);
                //var smo = new ProbabilisticCoordinateDescent(svm, trainingInputs, trainingOutputs);
                smo.Complexity = bestParameters[0].Value;
                double error = smo.Run();
                double validationError = smo.ComputeError(validationInputs, validationOutputs);
                return new Accord.MachineLearning.CrossValidationValues(svm, error, validationError);
            };

            var result = crossvalidation.Compute();
            double trainingErrors = result.Training.Mean;
            double validationErrors = result.Validation.Mean;

            ////var minError = result.Models.Select(y => y.ValidationValue).Min();
            ////var bestModel = result.Models.Where(x => x.ValidationValue == minError).FirstOrDefault();

            ////SupportVectorMachine bsvm = (SupportVectorMachine)bestModel.Model;


            ///* ------------------------------------------------------------------------------------------------------------------*/
            //Accord.MachineLearning.GridSearchRange[] rangesk =
            //{
            //    new Accord.MachineLearning.GridSearchRange("complexity" , new double[] {1E-3,1E-2,0.1}),
            //    new Accord.MachineLearning.GridSearchRange("gamma", new double[] {1E-4,0.001,0.01, 0.1}),
            //};
            //Console.Write("\nPerforming Grid Search on kSVM");
            //var gridsearchk = new Accord.MachineLearning.GridSearch<KernelSupportVectorMachine>(rangesk);
            //gridsearchk.Fitting = delegate (Accord.MachineLearning.GridSearchParameterCollection parametersk, out double errork)
            //{
            //    double complexity = parametersk["complexity"].Value;
            //    double gamma = parametersk["gamma"].Value;
            //    Gaussian gauss = new Gaussian();
            //    gauss.Gamma = gamma;
            //    KernelSupportVectorMachine ksvm = new KernelSupportVectorMachine(gauss, mat.Columns());
            //    SequentialMinimalOptimization ksmo = new SequentialMinimalOptimization(ksvm, trainDat, y_train);
            //    ksmo.Complexity = complexity;
            //    errork = ksmo.Run();
            //    return ksvm;
            //};

            //Console.Write("\nTraining Optimised kSVM...");
            //Accord.MachineLearning.GridSearchParameterCollection bestParametersk; double minErrork;
            //KernelSupportVectorMachine bksvm = gridsearchk.Compute(out bestParametersk, out minErrork);

            //var crossvalidationk = new Accord.MachineLearning.CrossValidation(size: mat.Length, folds: 10);
            //crossvalidationk.Fitting = delegate (int k, int[] indicesTrain, int[] indicesValidation)
            //{
            //    var trainingInputs = mat.Submatrix(indicesTrain);
            //    var trainingOutputs = labels.Submatrix(indicesTrain);
            //    var validationInputs = mat.Submatrix(indicesValidation);
            //    var validationOutputs = labels.Submatrix(indicesValidation);

            //    var ksvm = new SupportVectorMachine(mat.Columns());

            //    var ksmo = new SequentialMinimalOptimization(ksvm, trainingInputs, trainingOutputs);
            //    ksmo.Complexity = bestParametersk[0].Value;
            //    double error = ksmo.Run();
            //    double validationError = ksmo.ComputeError(validationInputs, validationOutputs);
            //    return new Accord.MachineLearning.CrossValidationValues(ksvm, error, validationError);
            //};

            //var resultk = crossvalidationk.Compute();
            //double trainingErrorsk = resultk.Training.Mean;
            //double validationErrorsk = resultk.Validation.Mean;

            ////double error = smo.Run();
            ////double errork = ksmo.Run();
            //Console.Write("\nTraining Complete");
            ////Console.Write("\nBest C: %0.2f", bestParameters[0].Value);
            ///* ------------------------------------------------------------------------------------------------------------------*/
            //Save SVM Model
            //System.Runtime.Serialization.Formatters.Binary.BinaryFormatter formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            FileStream Savestream = new FileStream("svm.bin", FileMode.Create);
            formatter.Serialize(Savestream, bsvm);
            Savestream.Close();
            Console.Write("\nSaved SVM Model");

            ////FileStream PCAStream = new FileStream("pca.bin", FileMode.Create);
            ////formatter.Serialize(PCAStream, pca);
            ////PCAStream.Close();
            ////Console.Write("\nSaved PCA matrix");

            //System.Runtime.Serialization.Formatters.Binary.BinaryFormatter formatterk = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
            //FileStream Savestreamk = new FileStream("ksvm.bin", FileMode.Create);
            //formatter.Serialize(Savestreamk, bksvm);
            //Savestreamk.Close();
            //Console.Write("\nSaved kSVM Model");

            //string tPath = String.Format(@"{0}\Pictorial\Files\svm.bin", Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData));
            //string tPath = "svm.bin"; //Local project location of SVM model 
            //Predict values

            for (int i = 0; i < nRowsTest; i++) //Uncomment for training                     
            {
                pred[i] = bsvm.Compute(testDat[i]); //Uncomment for training
            //    predksvm[i] = bksvm.Compute(testDat[i]);
            //    pred[i] = ksvm.Compute(testDat[i]);
            }
            ///* ------------------------------------------------------------------------------------------------------------------*/
            //// Use confusion matrix to compute some performance metrics
            int[] predict = new int[pred.Length];
            //int[] predictk = new int[predksvm.Length];
            for (int p = 0; p < pred.Length; p++)
            {
                predict[p] = Math.Sign(pred[p]);
            //    predictk[p] = Math.Sign(predksvm[p]);

            }
            ConfusionMatrix confusionMatrix = new ConfusionMatrix(predict, y_test, -1, 1); //Predicted, Expected, Positive, Negative Uncomment for training     
            //ConfusionMatrix confusionMatrixk = new ConfusionMatrix(predictk, y_test, -1, 1);

        }
    }
}
