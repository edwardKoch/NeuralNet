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

    // Destructor
    ~Matrix();

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
    void randomize(double_t min, double_t max);

    // Return a copy of the matrix
    Matrix<numRows, numCols> copy();

    // Transpose Matrix
    Matrix<numCols, numRows> transpose();

    // Perform Scalar Addition
    void add(double_t addor);

    // Perform Scalar Multiplication
    void multiply(double_t scalar);

    // Perform Elementwise Addition - Must be same shape
    void add(const Matrix<numRows, numCols>& addor);

    // Perform Elementwise Subtraction - Must be same shape
    void sub(const Matrix<numRows, numCols>& subtor);

    // Perform Elementwise Multiplication - Must be same shape
    void multiply(const Matrix<numRows, numCols>& scalar);

    // Matrix Multiplication 
    template<uint16_t otherCols>
    Matrix<numRows, otherCols> multiply(const Matrix<numCols, otherCols>& other);

    // Apply function to each element
    void applyFunction(double_t (*func)(double_t));

    // Print the matrix
    void print();

private:
    // Templated Matrix Friend
    template<uint16_t friendRows, uint16_t friendCols>
    friend class Matrix;

    // Random Number Generator
    std::mt19937 rng;

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
    : rng((uint32_t)std::time(0)),
      length(numRows * numCols)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] = 0.0;
    }
}

// Destructor
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::~Matrix()
{

}

// Get the value of an element
template<uint16_t numRows, uint16_t numCols>
inline double_t Matrix<numRows, numCols>::getElement(uint16_t row, uint16_t col) const
{ 
    uint16_t index = getIndex(row, col);
    if (index >= length)
    {
#if _DEBUG
        printf("Matrix - Get Element: Invalid Index\n");
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
        printf("Matrix - Set Element: Invalid Index\n");
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
inline void Matrix<numRows, numCols>::randomize(double_t min, double_t max)
{
    std::uniform_real_distribution<double_t> uniformDist(min, max);

    for (int i = 0; i < length; ++i)
    {
        matrix[i] = uniformDist(rng);
    }
}

// Return a copy of the matrix
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols> Matrix<numRows, numCols>::copy()
{
    Matrix<numRows, numCols> result;
    uint16_t index = 0;

    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            index = getIndex(row, col);
            result.setElement(row, col, matrix[index]);
        }
    }

    return result;
}

// Transpose Matrix
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numCols, numRows> Matrix<numRows, numCols>::transpose()
{
    Matrix<numCols, numRows> result;
    uint16_t index = 0;

    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            index = getIndex(row, col);
            result.setElement(col, row, matrix[index]);
        }
    }

    return result;
}

// Perform Scalar Addition
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::add(double_t addor)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] += addor;
    }
}

// Perform Scalar Multiplication
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::multiply(double_t scalar)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] *= scalar;
    }
}

// Perform Elementwise Addition - Must be same shape
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::add(const Matrix<numRows, numCols>& addor)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] += addor.matrix[i];
    }
}


// Perform Elementwise Subtraction - Must be same shape
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::sub(const Matrix<numRows, numCols>& subtor)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] -= subtor.matrix[i];
    }
}

// Perform Elementwise Multiplication - Must be same shape
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::multiply(const Matrix<numRows, numCols>& scalar)
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] *= scalar.matrix[i];
    }
}

// Matrix Multiplication 
template<uint16_t numRows, uint16_t numCols>
template<uint16_t otherCols>
inline Matrix<numRows, otherCols> Matrix<numRows, numCols>::multiply(const Matrix<numCols, otherCols>& other)
{
    Matrix<numRows, otherCols> result;
    uint16_t myIndex = 0;

    for (int resRow = 0; resRow < numRows; ++resRow)
    {
        for (int resCol = 0; resCol < otherCols; ++resCol)
        {
            double_t value = 0.0;
            // Sum the Dow Products of this rows and other cols
            for (int idx = 0; idx < numCols; ++idx)
            {
                myIndex = getIndex(resRow, idx);

                value += matrix[myIndex] * other.getElement(idx, resRow);
            }

            result.setElement(resRow, resCol, value);
        }
    }

    return result;
}

// Apply function to each element
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::applyFunction(double_t (*func)(double_t))
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] = func(matrix[i]);;
    }
}

// Print the matrix
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::print()
{
    for (int row = 0; row < numRows; ++row)
    {
        for (int col = 0; col < numCols; ++col)
        {
            uint16_t index = getIndex(row, col);

            printf("%.2f ", matrix[index]);
        }
        printf("\n");
    }
    printf("\n");
}

template<uint16_t numRows, uint16_t numCols>
inline uint16_t Matrix<numRows, numCols>::getIndex(uint16_t row, uint16_t col) const
{
    // Map 2D coordinates to 1D array index
    // ETK - INVESTIGATE ERROR
    return row * (numCols - 1) + col;
}

#endif