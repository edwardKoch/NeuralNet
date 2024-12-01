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

    // Copy Constructor
    Matrix(const Matrix<numRows, numCols>& m);

    // Destructor
    ~Matrix();

    // Copy Assignment
    Matrix<numRows, numCols>& operator=(const Matrix<numRows, numCols> other);

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
    for (int i = 0; i < length; ++i)
    {
        matrix[i] = 0.0;
    }
}

// Copy Constructor
template<uint16_t numRows, uint16_t numCols>
inline Matrix<numRows, numCols>::Matrix(const Matrix<numRows, numCols>& other)
{
    length = other.length;
    for (int i = 0; i < length; ++i)
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
inline Matrix<numRows, numCols>& Matrix<numRows, numCols>::operator=(const Matrix<numRows, numCols> other)
{
    if (this != &other)
    {
        length = other.length;
        for (int i = 0; i < length; ++i)
        {
            matrix[i] = other.matrix[i];
        }
    }
    return *this;
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

    for (int i = 0; i < length; ++i)
    {
        matrix[i] = uniformDist(rng);
    }
}

// Apply function to each element
template<uint16_t numRows, uint16_t numCols>
inline void Matrix<numRows, numCols>::applyFunction(double_t (*func)(double_t))
{
    for (int i = 0; i < length; ++i)
    {
        matrix[i] = func(matrix[i]);
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

            printf("%f ", matrix[index]);
        }
        printf("\n");
    }
    printf("\n");
}

template<uint16_t numRows, uint16_t numCols>
inline uint16_t Matrix<numRows, numCols>::getIndex(uint16_t row, uint16_t col) const
{
    // Map 2D coordinates to 1D array index
    return row * numCols + col;
}

#endif