#ifdef _WIN32
#include <windows.h>
HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
#else

#endif

#include "Matrix.h"
#include "NeuralNet.h"

// Global RNG
std::mt19937 rng(0);//(uint32_t)std::time(0));

int main()
{
    //////////////////////
    // Matrix Tests
    //////////////////////
    
    // 2x3 Matrix
    // 1 2 3
    // 4 5 6
    double_t m1Vals[6] = { 1, 2, 3, 4, 5, 6 };
    Matrix<2, 3> m1(m1Vals);
    m1.print();

    // 3x4 Matrix
    // 1  2  3  4
    // 5  6  7  8
    // 9 10 11 12
    double_t m2Vals[12] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    Matrix<3, 4> m2(m2Vals);
    m2.print();

    // Resulting Matrix should be 2x4 (from https://matrix.reshish.com/multCalculation.php)
    // 38  44  50  56
    // 83  98  113  128
    double_t m3Vals[8] = { 38, 44, 50, 56, 83, 98, 113, 128};
    Matrix<2, 4> m3 = m1.multiply(m2);

    m3.print();

    Matrix<4, 2> m4 = m3.transpose();
    m4.print();

    // Proof that there is ambiguity in Matrix Multiplication when the dimensions are the same
    Matrix<2, 2> m5({ 1, 2, 3, 4 });
    Matrix<2, 2> m6({ 1, 2, 3, 4 });

    m5.multiply(m6);

    m5.print();

    Matrix<2, 2> m7 = m5.multiply(m6);
    m7.print();


    //////////////////////
    // Neural Network
    //////////////////////
    
    // Random Number Generator
    std::uniform_real_distribution<float> uniformDist(0.0, 3.9);

    NeuralNet<2, 2, 2> brain(rng,    // Random Number Generator
        NN::Activations::SIGMOID, // Activation Function
        0.01); // Learning Rate

    double_t input[2] = { 0, 0 };
    double_t output[2] = { 0, 0 };
    double_t answer[2] = { -1, 1 };

    brain.randomize(-1.0, 1.0);

    brain.guess(input, output);

    printf("[%f, %f ]\n", output[0], output[1]);

    for (int i = 0; i < 100; ++i)
    {
        brain.train(input, answer);
    }

    brain.guess(input, output);

    printf("[%f, %f ]", output[0], output[1]);

}