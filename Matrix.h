//-----------------------------------------------------------------------------
// File: Matrix.h
// Author: Edward Koch
// Description: Holds the declaration of the Matrix Class
//              as created by following Daniel Schiffman's tutorial
// https://thecodingtrain.com/tracks/neural-networks/neural-networks/1-introduction//
//
// Revision History
// Author     Date        Description
//-----------------------------------------------------------------------------
// E. Koch    07/12/24    Initial Creation 
// E. Koch    11/30/24    Starting Fresh
//-----------------------------------------------------------------------------
#ifndef MATRIX_H
#define MATRIX_H

#include <cmath>
#include <math.h>
#include <random>
#include <stdint.h>
#include <stdio.h>

template<uint16_t numRows, uint16_t numCols>
class Matrix
{
public:
    // Constructor - initialize to 0
    Matrix();

    // Constructor - initialize from array
    Matrix(const double_t(&initArr)[numRows * numCols]);

    // Copy Constructor
    Matrix(const Matrix<numRows, numCols> &m);

    // Destructor
    ~Matrix();

    // Copy Assignment
    Matrix<numRows, numCols>& operator=(const Matrix<numRows, numCols> &other);

    // Fill the Matrix based on an array
    void fill(const double_t(&initArr)[numRows * numCols]);

    // Populate an array with the Matrix values
    void toArray(double_t (&arr)[numRows * numCols]);

    // Get the number of rows in the Matrix
    uint16_t getRows() const { return numRows; }

    // Get the number of rows in the Matrix
    uint16_t getCols() const { return numCols; }

    // Get the value of an element
    double_t getElement(uint16_t row, uint16_t col) const;

    // Set the value of an element
    void setElement(uint16_t row, uint16_t col, double_t value);

    // Set all values to 0
    void clear();

    // Randomize the values of the matrix given a range
    void randomize(std::mt19937 &rng, double_t min, double_t max);

    // Apply function to each element
    void applyFunction(double_t (*func)(double_t));

    // Print the matrix
    void print();

    // Scalar addition
    void add(double_t addor);

    // Element-wise addition
    void add(const Matrix<numRows, numCols> &addor);

    // Scalar subtraction
    void sub(double_t addor);

    // Element-wise subtraction
    void sub(const Matrix<numRows, numCols>& addor);

    // Scalar Multiplicaiton
    void multiply(double_t scalar);

    // Element-wise Multiplicaiton
    void multiply(const Matrix<numRows, numCols> &scalar);

    // Dot-Product Multiplication - Other must have the same number of rows as our columns
    template<uint16_t otherCols>
    Matrix<numRows, otherCols> matMultiply(const Matrix<numCols, otherCols> &other);

    // Transpose the Matrix
    Matrix<numCols, numRows> transpose();

private:
    // Templated Matrix Friend
    template<uint16_t friendRows, uint16_t friendCols>
    friend class Matrix;

    // Length of 1D array - Rows * Cols
    uint16_t length;

    // matrix representation - 1D array for memory access 
    double_t matrix[numRows * numCols];

    // Map 2D coordinates to 1D array index
    uint16_t getIndex(uint16_t row, uint16_t col) const;
};

// Constructor - initialize to 0
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::Matrix()
    : length(numRows * numCols)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = 0.0;
    }
}

// Constructor - initialize from array
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::Matrix(const double_t(&initArr)[numRows * numCols])
    : length(numRows* numCols)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = initArr[i];
    }
}

// Copy Constructor
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::Matrix(const Matrix<numRows, numCols> &other)
{
    length = other.length;
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = other.matrix[i];
    }
}

// Destructor
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::~Matrix()
{

}

// Copy Assignment
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>& Matrix<numRows, numCols>::operator=(const Matrix<numRows, numCols> &other)
{
    if (this != &other)
    {
        length = other.length;
        for (uint16_t i = 0; i < length; ++i)
        {
            matrix[i] = other.matrix[i];
        }
    }
    return *this;
}

// Fill the Matrix based on an array
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::fill(const double_t (&initArr)[numRows * numCols])
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = initArr[i];
    }
}

// Populate an array with the Matrix values
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::toArray(double_t (&arr)[numRows * numCols])
{
    for (uint16_t i = 0; i < length; ++i)
    {
        arr[i] = matrix[i];
    }
}

// Get the value of an element
template<uint16_t numRows, uint16_t numCols>
inline double_t Matrix<numRows, numCols>::getElement(uint16_t row, uint16_t col) const
{ 
    uint16_t index = getIndex(row, col);
    if (index >= length)
    {
#if _DEBUG
        printf("Matrix<%u, %u> - Get Element: Invalid Index %u (r%u, c%u)\n", numRows, numCols, index, row, col);
#endif
        return 0.0;
    }

    return matrix[index];
}


// Set the value of an element
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::setElement(uint16_t row, uint16_t col, double_t value)
{
    uint16_t index = getIndex(row, col);
    if (index >= length)
    {
#if _DEBUG
        printf("Matrix<%u, %u> - Set Element: Invalid Index %u (r%u, c%u)\n", numRows, numCols, index, row, col);
#endif
        return;
    }

    matrix[index] = value;
}

// Set all values to 0
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::clear()
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] = 0.0;
    }
}

// Randomize the values of the matrix given a range
template<uint16_t numRows, uint16_t numCols>
inline void  Matrix<numRows, numCols>::randomize(std::mt19937 &rng, double_t min, double_t max)
{
    std::uniform_real_distribution<double_t> uniformDist(min, max);

    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = uniformDist(rng);
    }
}

// Apply function to each element
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::applyFunction(double_t (*func)(double_t))
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] = func(matrix[i]);
    }
}

// Print the matrix
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::print()
{
    for (uint16_t row = 0; row < numRows; ++row)
    {
        for (uint16_t col = 0; col < numCols; ++col)
        {
            uint16_t index = getIndex(row, col);

            printf("%f ", matrix[index]);
        }
        printf("\n");
    }
    printf("\n");
}

// Scalar addition
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::add(double_t addor)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] += addor;
    }
}

// Element-wise addition
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::add(const Matrix<numRows, numCols> &addor)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] += addor.matrix[i];
    }
}

// Scalar subtraction
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::sub(double_t addor)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] -= addor;
    }
}

// Element-wise subtraction
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::sub(const Matrix<numRows, numCols>& addor)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] -= addor.matrix[i];
    }
}

// Scalar Multiplicaiton
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::multiply(double_t scalar)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] *= scalar;
    }
}

// Element-wise Multiplicaiton
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::multiply(const Matrix<numRows, numCols> &scalar)
{
    for (uint16_t i = 0; i < length; ++i)
    {
        matrix[i] *= scalar.matrix[i];
    }
}

// Dot-Product Multiplication - Other must have the same number of rows as our columns
template<uint16_t numRows, uint16_t numCols>
template<uint16_t otherCols>
inline Matrix<numRows, otherCols> Matrix<numRows, numCols>::matMultiply(const Matrix<numCols, otherCols> &other)
{
    // Self    Other       Result
    // 2x3     3x4         2x4
    // a b c   g h i j     (ag + bk + co) (ah + bl + cp) ... (aj + bn + cr)
    // d e f   k l m n     (dg + ek + fo) (dh + el + fp) ... (dj + en + fr)
    //         o p q r     Result by Index
    //                     (00,00 + 01,10 + 02,20) (00,01 + 01,11 + 02,21) ... (00,03 + 01,13 + 02,23)
    //                     (10,00 + 11,10 + 12,20) (10,01 + 11,11 + 12,21) ... (10,03 + 11,13 + 12,23)

    Matrix<numRows, otherCols> result;

    // Temp Variables
    double_t value = 0;
    uint16_t myIdx = 0;

    // For each row in the resulting Matrix
    for (uint16_t resRow = 0; resRow < numRows; ++resRow)
    {
        // For each col in the resulting Matrix
        for (uint16_t resCol = 0; resCol < otherCols; ++resCol)
        {
            // Initialize value to 0
            value = 0;

            // For each row/column pair in self and other
            for (uint16_t i = 0; i < numCols; ++i)
            {
                myIdx = getIndex(resRow, i);

                value += (matrix[myIdx] * other.getElement(i, resCol));
            }

            // Set value in result matrix
            result.setElement(resRow, resCol, value);
        }
    }

    return result;
}

// Transpose the Matrix
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numCols, numRows> Matrix<numRows, numCols>::transpose()
{
    Matrix<numCols, numRows> result;

    // For each row in the starting Matrix
    for (uint16_t row = 0; row < numRows; ++row)
    {
        // For each col in the starting Matrix
        for (uint16_t col = 0; col < numCols; ++col)
        {
            // Set the col,row of the result to the value in row,col of self
            result.setElement(col, row, getElement(row, col));
        }
    }

    return result;
}

template<uint16_t numRows, uint16_t numCols>
inline uint16_t Matrix<numRows, numCols>::getIndex(uint16_t row, uint16_t col) const
{
    // Map 2D coordinates to 1D array index
    return row * numCols + col;
}

#endif