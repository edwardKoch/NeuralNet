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
    // Random Number Generator
    std::mt19937 rng;
    std::uniform_real_distribution<float_t> uniformDist;

    // Learning Rate
    float_t learningRate;
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
NeuralNet<numInputs, numHidden, numOutputs>::NeuralNet(float_t learningRate)
    : rng((uint32_t)std::time(0)),
      uniformDist(-1.0, 1.0),
      learningRate(learningRate)
{

}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
NeuralNet<numInputs, numHidden, numOutputs>::~NeuralNet()
{

}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
std::vector<float_t> NeuralNet<numInputs, numHidden, numOutputs>::guess(std::vector<float_t> inputs)
{

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

