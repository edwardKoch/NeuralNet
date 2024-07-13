//-----------------------------------------------------------------------------
// File: NeuralNet.h
// Author: Edward Koch
// Description: Holds the declaration of the NeuralNetwork Class
//              as created by following Daniel Schiffman's tutorial
//  https://www.youtube.com/watch?v=XJ7HLz9VYz0&list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh
//
// Revision History
// Author     Date        Description
//-----------------------------------------------------------------------------
// E. Koch    07/11/24    Initial Creation 
//-----------------------------------------------------------------------------
#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "Perceptron.h"

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
class NeuralNet
{
public:
    NeuralNet(float_t learningRate);

    ~NeuralNet();

    std::vector<float_t> guess(std::vector<float_t> inputs);

    void train(std::vector<float_t> inputs, std::vector<float_t> targets);

    void print();

private:
    // Each Input Node has 1 input
    std::vector<Perceptron<1>> inputLayer;

    // Each Hidden Node takes in an input from each Input node
    std::vector<Perceptron<numInputs>> hiddenLayer;

    // Each Output Node takes in an input from each Hidden node
    std::vector<Perceptron<numHidden>> outputLayer;

    float_t learningRate;
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
NeuralNet<numInputs, numHidden, numOutputs>::NeuralNet(float_t learningRate)
    : learningRate(learningRate)
{
    inputLayer.reserve(numInputs);
    for (int i = 0; i < numInputs; ++i)
    {
        inputLayer.push_back(Perceptron(learningRate));
    }

    hiddenLayer.reserve(numHidden);
    for (int i = 0; i < numHidden; ++i)
    {
        hiddenLayer.push_back(Perceptron(learningRate));
    }

    outputLayer.reserve(numOutputs);
    for (int i = 0; i < numOutputs; ++i)
    {
        outputLayer.push_back(Perceptron(learningRate));
    }
}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
std::vector<float_t> NeuralNet<numInputs, numHidden, numOutputs>::guess(std::vector<float_t> inputs)
{
    // Initialize result to all 0s
    std::vector<float_t> result;
    for (int i = 0; i < numOutputs; ++i)
    {
        result.push_back(0.0);
    }

    if (inputs.size() != numInputs)
    {
        return result;
    }

    // Get the sum of all input nodes
    std::vector<float_t> inputLayerGuess;
    for (int i = 0; i < numInputs; ++i)
    {
        std::vector<float_t> input = { inputs[i] };
        inputLayerGuess.push_back(inputLayer[i].guess(input));
    }

    // Feed inputs forward to Hidden Layer
    std::vector<float_t> hiddenLayerGuess;
    for (int i = 0; i < numHidden; ++i)
    {
        hiddenLayerGuess.push_back(hiddenLayer[i].guess(inputLayerGuess));
    }

    // Get result from Output Layer
    for (int i = 0; i < numOutputs; ++i)
    {
        result[i] = outputLayer[i].guess(hiddenLayerGuess);
    }

    return result;
}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
void NeuralNet<numInputs, numHidden, numOutputs>::train(std::vector<float_t> inputs, std::vector<float_t> targets)
{

}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
void NeuralNet<numInputs, numHidden, numOutputs>::print()
{

}

#endif

