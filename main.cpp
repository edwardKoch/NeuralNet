#ifdef _WIN32
#include <windows.h>
HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
#else

#endif

#include "Matrix.h"
#include "NeuralNet.h"

// Global RNG
std::mt19937 rng(0);//(uint32_t)std::time(0));

// Run Matrix Math Tests
void matrixTest();

int main()
{
    matrixTest();

    // Random Number Generator
    std::uniform_real_distribution<float> uniformDist(0.0, 3.9);

    NeuralNet<2, 4, 2> brain(rng, NN::Activations::SIGMOID, 0.001);

}