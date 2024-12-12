#ifdef _WIN32
#include <windows.h>
HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
#else

#endif

#include "Matrix.h"
#include "NeuralNet.h"

#include "minstTest.h"

// Global RNG
std::mt19937 rng((uint32_t)std::time(0));

int main()
{
    //////////////////////
    // Matrix Tests
    //////////////////////
    // Removed in Previous Commit

    //////////////////////
    // Neural Network
    //////////////////////
    minstMain();
    
    /*
    // Random Number Generator
    std::uniform_real_distribution<float> uniformDist(0.0, 3.9);

    NeuralNet<2, 4, 2> brain(rng,    // Random Number Generator
        NN::Activations::SIGMOID, // Activation Function
        0.01); // Learning Rate

    // XOR Training Set
    double_t input[4][2] = {{ 0, 0 },
                            { 0, 1 },
                            { 1, 0 },
                            { 1, 1 } };

    // 2 Element answer where index 0 is true and index 1 is false
    double_t answer[4][2] = {{ 0, 1 },
                             { 1, 0 },
                             { 1, 0 },
                             { 0, 1 } };
    
    
    double_t output[2] = { 0, 0 };
    
    brain.randomize(-1.0, 1.0);

    printf("Before:\n");
    for (int i = 0; i < 4; ++i)
    {
        brain.guess(input[i % 4], output);
        printf("[%f, %f ]\n", output[0], output[1]);
    }

    brain.print();

    uint32_t numTraining = 600000;
    uint16_t batchSize = 100;
    uint32_t numCycles = numTraining / batchSize;

    for (uint32_t i = 0; i < numCycles; ++i)
    {
        brain.train(&input[0], &answer[0], 4, batchSize);

        double_t threshold = 0.05;

        double_t error = brain.test(&input[0], &answer[0], 4);

        if (error < threshold)
        {
            printf("Confident within %.03f with error %f after %d cycles", threshold, error, i);
            break;
        }

    }

    printf("\nAfter:\n");
    for (int i = 0; i < 4; ++i)
    {
        brain.guess(input[i % 4], output);
        printf("[%f, %f ]\n", output[0], output[1]);
    }

    brain.print();
    */
}