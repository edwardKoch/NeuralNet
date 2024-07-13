#include "NeuralNet.h"
int main()
{
    // Random Number Generator
    std::mt19937 rng((uint32_t)std::time(0));
    std::uniform_real_distribution<float> uniformDist(-1.0, 1.0);

    NeuralNet<2, 2, 1> brain(0.1);

}