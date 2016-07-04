using System;

namespace NeuralNet
{
    static class Util
    {
        /// <summary>
        /// Passes a value through the logistic function so that it has a smooth differentiable value.
        /// </summary>
        /// <param name="value">The value we are squashing</param>
        /// <returns>Returns the squashed value.</returns>
        public static double Sigmoid(double value)
        {
            return (1 / (1 + Math.Exp(-value)));
        }

        /// <summary>
        /// Applies the logistic function to every element in a matrix.
        /// </summary>
        /// <param name="matrix">The matrix to perform the operation on.</param>
        /// <returns>Returns a matrix squashed values.</returns>
        public static Matrix2D Sigmoid(Matrix2D matrix)
        {
            Matrix2D result = new Matrix2D(matrix.RowCount, matrix.ColumnCount);

            for (int m = 0; m < matrix.RowCount; m++)
            {
                for (int n = 0; n < matrix.ColumnCount; n++)
                {
                    result[m, n] = Sigmoid(matrix[m, n]);
                }
            }

            return result;
        }

        /// <summary>
        /// The derivative of the logistic function.
        /// </summary>
        /// <param name="value">The value we are computing the derivative with.</param>
        /// <returns>Returns the derivative.</returns>
        public static double SigmoidDeriv(double value)
        {
            return value * (1 - value);
        }

        /// <summary>
        /// Applies the derivative of the logisitc function to every element in a matrix.
        /// </summary>
        /// <param name="matrix">The matrix to perform the calculation on.</param>
        /// <returns></returns>
        public static Matrix2D SigmoidDeriv(Matrix2D matrix)
        {
            Matrix2D result = new Matrix2D(matrix.RowCount, matrix.ColumnCount);

            for (int m = 0; m < matrix.RowCount; m++)
            {
                for (int n = 0; n < matrix.ColumnCount; n++)
                {
                    result[m, n] = SigmoidDeriv(matrix[m, n]);
                }
            }

            return result;
        }

        /// <summary>
        /// Calculates examples of all NeuralNetUtility functions to test if they perform correctly.
        /// </summary>
        public static void TestUtilityFunctions()
        {
            /// Test logistic function and its derivative.
            double logisticValue = 5.6836;
            Console.WriteLine("Logistic function on " + logisticValue + " = " + Sigmoid(logisticValue));
            double logisticDerivValue = Sigmoid(logisticValue);
            Console.WriteLine("Logistic function derivative on " + logisticDerivValue + " = " + SigmoidDeriv(logisticDerivValue));
            Console.WriteLine();
        }

        /// <summary>
        /// Function to test the Matrix2D class to make sure it performs correctly.
        /// </summary>
        public static void TestMatrix2D()
        {
            /// Two arbitrary matrices to test on.
            double[,] A = new double[,]
            {
                { 1, 2, 3 },
                { 4, 5, 6 }
            };

            double[,] B = new double[,]
            {
                { 1, 2 },
                { 3, 4 },
                { 5, 6 }
            };

            double[,] constant = new double[,]
            {
                { 2, 2, 2 },
                { 2, 2, 2 }
            };

            /// Test matrix to string.
            Matrix2D matrixA = new Matrix2D(A);
            Matrix2D matrixB = new Matrix2D(B);
            Console.WriteLine("Matrix A\n" + matrixA);
            Console.WriteLine("Matrix B\n" + matrixB);

            /// Test randomize matrix.
            Matrix2D matrixC = new Matrix2D(5, 5);
            matrixC.Randomize(-1.0f, 1.0f, new Random());
            Console.WriteLine("Random Matrix C\n" + matrixC);

            /// Test matrix multiplication.
            Matrix2D matrixD = Matrix2D.MultiplyMatrices(matrixA, matrixB);
            Console.WriteLine("Matrix D = AB\n" + matrixD);

            /// Test transpose matrix.
            Console.WriteLine("Transpose of A\n" + matrixA.Transpose());

            /// Test of matrix operators.
            Matrix2D constantMatrix = new Matrix2D(constant);
            Console.WriteLine("A - 2\n" + (matrixA - constantMatrix));
            Console.WriteLine("A + 2\n" + (matrixA + constantMatrix));
            Console.WriteLine("A * Matrix of all 2's\n" + (matrixA * constantMatrix));
            Console.WriteLine("3 * A\n" + (3 * matrixA));
            Console.WriteLine("A * 3\n" + (matrixA * 3));

            /// Test absolute mean of matrix.
            Console.WriteLine("Absolute mean of constant matrix = " + constantMatrix.AbsoluteMean());
            Console.WriteLine("Absolute mean of matrix A = " + matrixA.AbsoluteMean());
        }
    }
}