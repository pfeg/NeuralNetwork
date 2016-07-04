using System;

namespace NeuralNet
{
    /// <summary>
    /// A representation of a two dimensional matrix.
    /// </summary>
    class Matrix2D
    {
        /// <summary>
        /// The number of rows in the matrix.
        /// </summary>
        private int rowCount;

        /// <summary>
        /// The number of columns in the matrix.
        /// </summary>
        private int columnCount;

        /// <summary>
        /// A two dimensional array of doubles that we will
        /// use to represent our matrix.
        /// </summary>
        private double[,] matrix;

        /// <summary>
        /// Creates a Matrix2D with the specified number of rows and columns.
        /// </summary>
        /// <param name="rowCount">The number of rows in the matrix.</param>
        /// <param name="columnCount">The number of columns in the matrix.</param>
        public Matrix2D(int rowCount, int columnCount)
        {
            this.rowCount = rowCount;
            this.columnCount = columnCount;
            matrix = new double[rowCount, columnCount];
        }

        /// <summary>
        /// Creates a Matrix2D from a two dimensional array of doubles.
        /// </summary>
        /// <param name="matrix">The double array to create the Matrix2D from.</param>
        public Matrix2D(double[,] matrix)
        {
            rowCount = matrix.GetLength(0);
            columnCount = matrix.GetLength(1);
            this.matrix = new double[rowCount, columnCount];

            Array.Copy(matrix, this.matrix, matrix.Length);
        }

        public int RowCount
        {
            get { return rowCount; }
        }

        public int ColumnCount
        {
            get { return columnCount; }
        }

        public double this[int rowNum, int columnNum]
        {
            get { return matrix[rowNum, columnNum]; }
            set { matrix[rowNum, columnNum] = value; }
        }

        /// <summary>
        /// Calculates the mean of the absolute values of every value in the matrix.
        /// </summary>
        /// <returns>Returns the absolute mean of the matrix.</returns>
        public double AbsoluteMean()
        {
            double result = 0;

            for (int m = 0; m < rowCount; m++)
            {
                for (int n = 0; n < columnCount; n++)
                {
                    result += Math.Abs(matrix[m, n]);
                }
            }

            return result / (rowCount * columnCount);
        }

        /// <summary>
        /// Gets the specified row as a double array.
        /// </summary>
        /// <param name="rowNum">The row number to return. Must be within the bounds of the matrix.</param>
        /// <returns>Returns the specified row.</returns>
        public double[] GetRow(int rowNum)
        {
            double[] row = new double[columnCount];

            for (int n = 0; n < columnCount; n++)
            {
                row[n] = matrix[rowNum, n];
            }

            return row;
        }

        /// <summary>
        /// Gets the specified columnn as a double array.
        /// </summary>
        /// <param name="columnNum">The row number to return. Must be within the bounds of the matrix.</param>
        /// <returns>Returns the specified column.</returns>
        public double[] GetColumn(int columnNum)
        {
            double[] column = new double[rowCount];

            for (int m = 0; m < rowCount; m++)
            {
                column[m] = matrix[m, columnNum];
            }

            return column;
        }

        /// <summary>
        /// Randomizes the values of the matrix.
        /// </summary>
        /// <param name="min">The minimum random value.</param>
        /// <param name="max">The maximum random value.</param>
        /// <param name="seed">The seed for random values.</param>
        public void Randomize(double min, double max, Random random)
        {
            double range = max - min;

            for (int m = 0; m < rowCount; m++)
            {
                for (int n = 0; n < columnCount; n++)
                {
                    matrix[m, n] = range * random.NextDouble() + min;
                }
            }
        }

        /// <summary>
        /// Transposes a matrix.
        /// </summary>
        public Matrix2D Transpose()
        {
            Matrix2D transposeMatrix = new Matrix2D(columnCount, rowCount);

            for (int m = 0; m < rowCount; m++)
            {
                for (int n = 0; n < columnCount; n++)
                {
                    transposeMatrix[n, m] = matrix[m, n];
                }
            }

            return transposeMatrix;
        }

        public static Matrix2D MultiplyMatrices(Matrix2D matrix1, Matrix2D matrix2)
        {
            int resultRows = matrix1.RowCount;
            int resultColumns = matrix2.ColumnCount;
            Matrix2D result = new Matrix2D(resultRows, resultColumns);

            for (int m = 0; m < resultRows; m++)
            {
                for (int n = 0; n < resultColumns; n++)
                {
                    result[m, n] = Dot(matrix1.GetRow(m), matrix2.GetColumn(n));
                }
            }

            return result;
        }

        public static Matrix2D operator *(Matrix2D matrix1, Matrix2D matrix2)
        {
            Matrix2D result = new Matrix2D(matrix1.RowCount, matrix1.ColumnCount);

            for (int m = 0; m < result.RowCount; m++)
            {
                for (int n = 0; n < result.ColumnCount; n++)
                {
                    result[m, n] = matrix1[m, n] * matrix2[m, n];
                }
            }

            return result;
        }

        public static Matrix2D operator *(double constant, Matrix2D matrix)
        {
            Matrix2D result = new Matrix2D(matrix.RowCount, matrix.ColumnCount);

            for (int m = 0; m < result.RowCount; m++)
            {
                for (int n = 0; n < result.ColumnCount; n++)
                {
                    result[m, n] = matrix[m, n] * constant;
                }
            }

            return result;
        }

        public static Matrix2D operator *(Matrix2D matrix, double constant)
        {
            return constant * matrix;
        }

        public static Matrix2D operator +(Matrix2D matrix1, Matrix2D matrix2)
        {
            Matrix2D result = new Matrix2D(matrix1.RowCount, matrix1.ColumnCount);

            for (int m = 0; m < result.RowCount; m++)
            {
                for (int n = 0; n < result.ColumnCount; n++)
                {
                    result[m, n] = matrix1[m, n] + matrix2[m, n];
                }
            }

            return result;
        }

        public static Matrix2D operator -(Matrix2D matrix1, Matrix2D matrix2)
        {
            Matrix2D result = new Matrix2D(matrix1.RowCount, matrix1.ColumnCount);

            for (int m = 0; m < result.RowCount; m++)
            {
                for (int n = 0; n < result.ColumnCount; n++)
                {
                    result[m, n] = matrix1[m, n] - matrix2[m, n];
                }
            }

            return result;
        }

        /// <summary>
        /// Creates a string representation of the matrix.
        /// </summary>
        /// <returns>Returns the string representation of the matrix.</returns>
        public override string ToString()
        {
            string output = "";

            for (int m = 0; m < rowCount; m++)
            {
                for (int n = 0; n < columnCount; n++)
                {
                    output += matrix[m, n] + "   ";
                }
                output += '\n';
            }

            return output;
        }

        /// <summary>
        /// Calculates the dot product of two vectors.
        /// </summary>
        /// <param name="vector1">The first vector.</param>
        /// <param name="vector2">The second vector.</param>
        /// <returns>Returns the dot product.</returns>
        private static double Dot(double[] vector1, double[] vector2)
        {
            double result = 0;

            for (int i = 0; i < vector1.Length; i++)
            {
                result += vector1[i] * vector2[i];
            }

            return result;
        }
    }
}