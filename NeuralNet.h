//-----------------------------------------------------------------------------
// File: NeuralNet.h
// Author: Edward Koch
// Description: Holds the declaration of the NeuralNetwork Class
//              as created by following Daniel Schiffman's tutorial
// https://thecodingtrain.com/tracks/neural-networks/neural-networks/1-introduction//
//
// Revision History
// Author     Date        Description
//-----------------------------------------------------------------------------
// E. Koch    07/11/24    Initial Creation 
//-----------------------------------------------------------------------------
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <ctime>
#include <math.h>
#include <random>
#include <stdint.h>
#include <vector>

namespace NN
{
    enum class Activations : uint8_t
    {
        SIGMOID,
        RELU
    };

    double_t sigmoid(double_t input)
    {
        // Sigmoid Approximation
        return (input / (1 + abs(input)));
    }

    double_t sigmoidDerivative(double_t input)
    {
        double_t sigInput = sigmoid(input);
        return (sigInput * (1 - sigInput));
    }

    double_t relu(double_t input)
    {
        if (input < 0.0)
        {
            return 0;
        }
        else
        {
            return input;
        }
    }

    double_t reluDerivative(double_t input)
    {
        if (input < 0.0)
        {
            return 0;
        }
        else
        {
            return 1;
        }
    }


    // Round a double to prevent precision errors
    double_t round(double_t intput)
    {
        return std::round(intput * 1000.0) / 1000.0;
    }

    double_t square(double_t input)
    {
        return input * input;
    }

    double_t invert(double_t input)
    {
        return 1.0 / input;
    }
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
class NeuralNet
{
public:
    
    NeuralNet(std::mt19937 rngIn, 
              NN::Activations activation = NN::Activations::SIGMOID, 
              double_t learningRate = 0.001);

    ~NeuralNet();

    // Set the Learning Rate
    void setLearningRate(double_t lr);

private:
    // Random Number Generator
    std::mt19937 rng;
    std::uniform_real_distribution<double_t> uniformDist;

    // Activation Function to use
    NN::Activations activationFunciton;

    // Activation Function
    double_t(*actFunct)(double_t);
    double_t(*actFunctDeriv)(double_t);

    // Learning Rate
    double_t learningRate;

};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline NeuralNet<numInputs, numHidden, numOutputs>::NeuralNet(std::mt19937 rngIn, NN::Activations activation, double_t learningRate)
    : rng(rngIn),
      uniformDist(-1.0, 1.0),
      activationFunciton(activation),
      actFunct(0),
      learningRate(learningRate)
{
    // Choose Activation Function
    switch (activationFunciton)
    {
    case NN::Activations::SIGMOID:
        actFunct = NN::sigmoid;
        actFunctDeriv = NN::sigmoidDerivative;
        break;

    case NN::Activations::RELU:
        actFunct = NN::relu;
        actFunctDeriv = NN::reluDerivative;
        break;

    default:
        actFunct = NN::sigmoid;
        actFunctDeriv = NN::sigmoidDerivative;
        break;
    }
}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline NeuralNet<numInputs, numHidden, numOutputs>::~NeuralNet()
{

}

// Set the Learning Rate
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::setLearningRate(double_t lr)
{
    learningRate = lr;
}
#endif

