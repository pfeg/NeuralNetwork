using System;
using NeuralNet;

namespace SimpleNeuralNet
{
    class Program
    {
        /// Delegates type definitions for the functions to get
        /// input and output to train the neural network.
        public delegate Matrix2D Inputs(int batchSize, Random random);
        public delegate Matrix2D Outputs(Matrix2D inputs);

        static void Main(string[] args)
        {
            /// Value the error derivative is multiplied by to change how fast the network weights change.
            double alpha = .1;
            /// How many times we will loop the training code.
            int iterations = 10000;
            /// How often we will output results.
            int outputFrequency = iterations / 1;
            /// The seed for the random values.
            int randomSeed = 7;
            /// The number of neurons in the hidden layer.
            int hiddenNeurons = 50;
            /// The number of input neurons.
            int inputSize = 1;
            /// The number of output neurons.
            int outputSize = 1;
            /// The number of input sets calculated every iteration.
            int batchSize = 50;
            /// Seeded random number generator.
            Random random = new Random(randomSeed);
            /// Delegates for getting inputs and outputs to train the network.
            Inputs trainingInputs = GetTrigInputs;
            Outputs trainingOutputs = GetTrigOutputs;
            
            /// Initialize this with random values with a mean of zero.
            Matrix2D syn0 = new Matrix2D(inputSize, hiddenNeurons);
            syn0.Randomize(-1.0f, 1.0f, random);
            Matrix2D syn1 = new Matrix2D(hiddenNeurons, outputSize);
            syn1.Randomize(-1.0f, 1.0f, random);

            /// The first layer of neurons, the input layer.
            Matrix2D l0 = new Matrix2D(batchSize, inputSize);
            /// The second layer of neurons, the hidden layer.
            Matrix2D l1 = new Matrix2D(batchSize, hiddenNeurons);
            /// The third layer of neurons, the output layer.
            Matrix2D l2 = new Matrix2D(batchSize, outputSize);

            /// The total error of every for every input set.
            Matrix2D l1_error = new Matrix2D(batchSize, hiddenNeurons);
            Matrix2D l2_error = new Matrix2D(batchSize, outputSize);

            /// The change in weights.
            Matrix2D l1_delta = new Matrix2D(batchSize, hiddenNeurons);
            Matrix2D l2_delta = new Matrix2D(batchSize, outputSize);

            Console.WriteLine("Training Neural Network...\n");

            /// The main loop.
            /// This is where we will train our network.
            for (int i = 0; i < iterations; i++)
            {
                Matrix2D inputs = trainingInputs(batchSize, random);
                Matrix2D outputs = trainingOutputs(inputs);

                /// Forward Propogation.
                l0 = inputs;
                l1 = Util.Sigmoid(Matrix2D.MultiplyMatrices(l0, syn0));
                l2 = Util.Sigmoid(Matrix2D.MultiplyMatrices(l1, syn1));

                /// Calculate the squared error of the last layer.
                l2_error = 0.5f * ((outputs - l2) * (outputs - l2));

                /// Calculate the error weighted derivative of the last layer.
                l2_delta = (l2 - outputs) * Util.SigmoidDeriv(l2);

                /// Calculate the error of the hidden layer.
                l1_error = Matrix2D.MultiplyMatrices(l2_delta, syn1.Transpose());

                /// Calculate the error weighted derivative of the hidden layer.
                l1_delta = l1_error * Util.SigmoidDeriv(l1);

                /// Update weights.
                syn1 -= alpha * Matrix2D.MultiplyMatrices(l1.Transpose(), l2_delta);
                syn0 -= alpha * Matrix2D.MultiplyMatrices(l0.Transpose(), l1_delta);

                /// Output the predicted results after outputFrequency training iterations.
                if (i % outputFrequency == (outputFrequency - 1))
                {
                    Console.WriteLine("Predicted results after training " + (i + 1) + " iterations.");
                    for (int k = 0; k < batchSize; k++)
                    {
                        Console.WriteLine("Predicted: " + l2[k, 0] + "     Actual: " + trainingOutputs(inputs)[k, 0]);
                    }
                    Console.WriteLine("Average Squared Error: " + l2_error.AbsoluteMean());
                    Console.WriteLine();
                }
            }

            Console.WriteLine("{0} training iterations finished.\n", iterations);
            Console.WriteLine("Testing Neural Network...");

            /// Test the neural network against additional examples to see how well it performs.
            int testCount = 10000;
            double averageError = 0;
            Random testRandom = new Random();
            for (int i = 0; i < testCount / batchSize; i++)
            {
                Matrix2D inputs = trainingInputs(batchSize, testRandom);

                l0 = inputs;
                l1 = Util.Sigmoid(Matrix2D.MultiplyMatrices(l0, syn0));
                l2 = Util.Sigmoid(Matrix2D.MultiplyMatrices(l1, syn1));

                Matrix2D outputs = trainingOutputs(inputs);

                for (int m = 0; m < outputs.RowCount; m++)
                {
                    averageError += Math.Abs(outputs[m, 0] - l2[m, 0]);
                }
            }

            Console.WriteLine("Testing complete.\n");
            Console.WriteLine("The neural network had an average error of {0}.\n", averageError / testCount);
        }

        /// <summary>
        /// Generates a set of inputs to feed into the nerual network.
        /// </summary>
        /// <param name="batchSize">How many input sets are in one matrix.</param>
        /// <returns>Returns a matrix of inputs.</returns>
        private static Matrix2D GetAdditionInputs(int batchSize, Random random)
        {
            Matrix2D inputs = new Matrix2D(batchSize, 3);
            const int max = 100;
            const int spread = 50;

            /// For every set of inputs, generate three integer values.
            for (int m = 0; m < batchSize; m++)
            {
                int randomInt = random.Next(10);
                /// 30% chance to generate three random integers.
                if (randomInt >= 7)
                {
                    inputs[m, 0] = random.Next(max);
                    inputs[m, 1] = random.Next(max);
                    inputs[m, 2] = random.Next(max);
                }
                /// 40% chanve to generate two random integers that add up to the third integer.
                else if (randomInt <= 3)
                {
                    inputs[m, 0] = random.Next(max);
                    inputs[m, 1] = random.Next(max);
                    inputs[m, 2] = inputs[m, 0] + inputs[m, 1];
                }
                /// 30% chance to generate two random integers that almost add up to the third integer.
                else
                {
                    inputs[m, 0] = random.Next(max);
                    inputs[m, 1] = random.Next(max);
                    inputs[m, 2] = inputs[m, 0] + inputs[m, 1] + (random.Next(spread) - (spread / 2));
                }

                //double largest = Math.Max(Math.Max(inputs[m, 0], inputs[m, 1]), inputs[m, 2]);
                //inputs[m, 0] /= largest;
                //inputs[m, 1] /= largest;
                //inputs[m, 2] /= largest;
            }

            return inputs;
        }

        /// <summary>
        /// Generates the correct set of outputs for a given input set.
        /// </summary>
        /// <param name="inputs">The inputs to calculate the outputs from.</param>
        /// <returns>Returns a correct matrix of outputs.</returns>
        private static Matrix2D GetAdditionOutputs(Matrix2D inputs)
        {
            Matrix2D outputs = new Matrix2D(inputs.RowCount, 1);

            for (int m = 0; m < outputs.RowCount; m++)
            {
                if ((inputs[m, 0] + inputs[m, 1]) == inputs[m, 2])
                {
                    outputs[m, 0] = 1;
                }
                else
                {
                    outputs[m, 0] = 0;
                }
            }

            return outputs;
        }

        private static Matrix2D TestGetInputs(int batchSize)
        {
            /// Simple matrix of inputs.
            double[,] input = new double[,]
            {
                { 0, 0, 1 },
                { 1, 1, 1 },
                { 1, 0, 1 },
                { 0, 1, 1 }
            };
            return new Matrix2D(input);
        }

        private static Matrix2D TestGetOutputs(Matrix2D inputs)
        {
            /// Simple matrix of outputs.
            double[,] output = new double[,]
            {
                { 0 },
                { 0 },
                { 1 },
                { 1 }
            };
            return new Matrix2D(output);
        }

        private static Matrix2D GetTrigInputs(int batchSize, Random random)
        {
            Matrix2D inputs = new Matrix2D(batchSize, 1);

            for (int m = 0; m < inputs.RowCount; m++)
            {
                inputs[m, 0] = random.NextDouble() * Math.PI * 2;
            }

            return inputs;
        }

        private static Matrix2D GetTrigOutputs(Matrix2D inputs)
        {
            Matrix2D outputs = new Matrix2D(inputs.RowCount, 1);

            for (int m = 0; m < outputs.RowCount; m++)
            {
                outputs[m, 0] = (Math.Sin(inputs[m, 0]) / 2f) + 0.5f;
            }

            return outputs;
        }
    }
}