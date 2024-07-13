#include "Perceptron.h"
#include "NeuralNet.h"
int main()
{
    // Random Number Generator - included from Perceptron
    std::mt19937 rng((uint32_t)std::time(0));
    std::uniform_real_distribution<float> uniformDist(-1.0, 1.0);

    Perceptron<2> binaryP(0.1);

    // Sample Training Set
    uint16_t correct = 0;
    uint16_t total = 1000;

    float_t x = 0.0;
    float_t y = 0.0;
    float_t label = 0.0;

    for (int i = 0; i < total; ++i)
    {
        x = uniformDist(rng);
        y = uniformDist(rng);

        std::vector<float_t> traingSet = { x, y };
        if (x > y)
        {
            label = 1;
        }
        else
        {
            label = -1;
        }

        binaryP.train(traingSet, label);

        // Test Perceptron
        x = uniformDist(rng);
        y = uniformDist(rng);
        std::vector<float_t> testSet = { x, y };
        if (x > y)
        {
            label = 1;
        }
        else
        {
            label = -1;
        }
        float_t guess = binaryP.guess(testSet);

        //printf("Guess: %.6f, Label: %.6f\n", guess, label);

        if (guess - label < 0.000005)
        {
            printf("Correct!\n");
            ++correct;
        }
        else
        {
            printf("FAIL! Error: %.6f, Inputs: %.2f, %.2f\n", guess - label, x, y);
        }
    }

    printf("Success Rate: %.2f%%\n", float_t(correct) / (total) * 100);
    binaryP.print();
}