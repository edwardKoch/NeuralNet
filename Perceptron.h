//-----------------------------------------------------------------------------
// File: NeuralNet.h
// Author: Edward Koch
// Description: Holds the declaration of the Perceptron Class
//              as created by following Daniel Schiffman's tutorial
// https://thecodingtrain.com/tracks/neural-networks/neural-networks/1-introduction//
// Revision History
// Author     Date        Description
//-----------------------------------------------------------------------------
// E. Koch    07/11/24    Initial Creation 
//-----------------------------------------------------------------------------
#ifndef PERCEPTRON_H
#define PERCEPTRON_H

#include <ctime>
#include <math.h>
#include <random>
#include <stdint.h>
#include <vector>

template<uint16_t numInputs>
class Perceptron
{
public:
    Perceptron(float_t learningRate);

    ~Perceptron();

    float_t guess(std::vector<float_t> inputs);

    void train(std::vector<float_t> inputs, float_t target);

    void print();

private:
    // Random Number Generator
    std::mt19937 rng;
    std::uniform_real_distribution<float_t> uniformDist;

    // Learning Rate
    float_t learningRate;

    // Input Weights
    std::vector<float_t> weights;

    // Bias
    float_t bias;

};

template<uint16_t numInputs>
Perceptron<numInputs>::Perceptron(float_t learningRate)
    : rng((uint32_t)std::time(0)),
      uniformDist(-1.0, 1.0),
      learningRate(learningRate)
{
    weights.reserve(numInputs);

    // Initialize Weights and Bias to random normalized values
    for (int i = 0; i < numInputs; ++i)
    {
        weights.push_back(uniformDist(rng));
    }

    bias = uniformDist(rng);
}

template<uint16_t numInputs>
Perceptron<numInputs>::~Perceptron()
{

}

template<uint16_t numInputs>
float_t Perceptron<numInputs>::guess(std::vector<float_t> inputs)
{
    if (inputs.size() != numInputs)
    {
        return 0;
    }

    double_t sum = 0.0;

    // Sum the weighted inputs 
    for (int i = 0; i < numInputs; ++i)
    {
        sum += inputs[i] * weights[i];
    }

    // Add Bias
    sum += bias;

    // TODO - Activation Function (temporarily sign)
    if (sum > 0)
    {
        return 1.0;
    }
    else
    {
        return -1.0;
    }
}

template<uint16_t numInputs>
void Perceptron<numInputs>::train(std::vector<float_t> inputs, float_t target)
{
    if (inputs.size() != numInputs)
    {
        return;
    }

    // Guess an output based on inputs
    float_t output = guess(inputs);

    // Calculate error from target
    float_t error = target - output;

    // Tune weights based on error and input (dampened by LR)
    for (int i = 0; i < numInputs; ++i)
    {
        weights[i] += (error * inputs[i]) * learningRate;
    }

    // Tune Bias
    bias += error * learningRate;

}

template<uint16_t numInputs>
void Perceptron<numInputs>::print()
{
    for (int i = 0; i < numInputs; ++i)
    {
        printf("%.2f, ", weights[i]);
    }
    printf("\n%.2f\n", bias);
}

#endif