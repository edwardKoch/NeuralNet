#include "Matrix.h"
#include "NeuralNet.h"


int main()
{
    // Random Number Generator
    std::mt19937 rng((uint32_t)std::time(0));
    std::uniform_real_distribution<float> uniformDist(-1.0, 1.0);

    NeuralNet<2, 2, 1> brain(0.1);

    // Matrix Testing
    Matrix<2, 3> m1;
    Matrix<2, 3> m2;
    Matrix<3, 2> m3;
    m1.print(); // Should be 2x3 0s
    m1.add(1);
    m2.add(2);
    m3.add(0.5);
    m1.print(); // Should be 2x3 1s
    m1.multiply(2);
    m1.print(); // Should be 2x3 2s
    m1.add(m2);
    m1.print(); // Should be 2x3 4s
    Matrix<2, 2> m4 = m1.multiply(m3);
    m3.print(); // Should be 3x2 0.5s
    m4.print(); // Should be 2x2 6s
    m2.setElement(0, 1, 3);
    m2.print(); // Should be 2x3 2 3 2, 2 2 2
    m3 = m2.transpose();
    m3.print(); // Should be 3x2 2 2, 3 2, 2 2
    Matrix<3, 3> m5;
    m5.randomize(-5, 5);
    m5.print(); // Should be 3x3 with random values
}