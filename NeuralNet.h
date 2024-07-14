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
        SIGMOID
    };

    double_t sigmoid(double_t input)
    {
        // Sigmoid Approximation
        return (input / (1 + abs(input)));
    }
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
class NeuralNet
{
public:
    
    NeuralNet(NN::Activations activation = NN::Activations::SIGMOID, double_t learningRate = 0.1);

    ~NeuralNet();

    // Guess an output vector based on an input vector
    std::vector<double_t> guess(const std::vector<double_t>& inputs);

    // Train the neural net based on an input vector and a target vector
    void train(const std::vector<double_t>& inputs, const std::vector<double_t>& targets);

    // Print the weights and Bias for the entire network
    void print();

private:
    // Random Number Generator
    std::mt19937 rng;
    std::uniform_real_distribution<double_t> uniformDist;

    // Activation Function to use
    NN::Activations activationFunciton;

    // Activation Function
    double_t(*actFunct)(double_t);

    // Learning Rate
    double_t learningRate;

    // Input Matrix - store vector input values
    Matrix<numInputs, 1> inputValues;

    // Weight Matrix - Input to Hidden
    Matrix<numHidden, numInputs> inputWeights;

    // Bias matrix - Hidden
    Matrix<numHidden, 1> hiddenBias;

    // Hidden Layer Output - Matrix Product of Input Values and Input Weights/Bias
    Matrix<numHidden, 1> hiddenOutput;

    // Weight Matrix - Hidden to output
    Matrix<numOutputs, numHidden> outputWeights;

    // Bias Matrix - Output
    Matrix<numOutputs, 1> outputBias;

    // Output Matrix - Matrix Product of HiddenOutput and Output Weights/Bias
    Matrix<numOutputs, 1> outputValues;

    // Output Vector - From matrix to output
    std::vector<double_t> outputVector;

    // Target Matrix - used in training
    Matrix<numOutputs, 1> targetMatrix;

    // Error Matrix from output - used in training
    Matrix<numOutputs, 1> outputError;

    // Error Matrix from Hidden Layer - used in training
    Matrix<numHidden, 1> hiddenError;
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline NeuralNet<numInputs, numHidden, numOutputs>::NeuralNet(NN::Activations activation, double_t learningRate)
    : rng((uint32_t)std::time(0)),
      uniformDist(-1.0, 1.0),
      activationFunciton(activation),
      actFunct(0),
      learningRate(learningRate),
      inputValues(),
      inputWeights(),
      hiddenBias(),
      hiddenOutput(),
      outputWeights(),
      outputBias(),
      outputValues(),
      outputVector(numOutputs),
      targetMatrix(),
      outputError(),
      hiddenError()
{
    inputWeights.randomize(-1.0, 1.0);
    hiddenBias.randomize(-1.0, 1.0);

    outputWeights.randomize(-1.0, 1.0);
    outputBias.randomize(-1.0, 1.0);

    // Choose Activation Function
    switch (activationFunciton)
    {
    case NN::Activations::SIGMOID:
        actFunct = NN::sigmoid;
        break;

    default:
        actFunct = NN::sigmoid;
        break;
    }
}

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline NeuralNet<numInputs, numHidden, numOutputs>::~NeuralNet()
{

}
// Guess an output vector based on an input vector
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline std::vector<double_t> NeuralNet<numInputs, numHidden, numOutputs>::guess(const std::vector<double_t>& inputs)
{
    // Clear Output
    outputVector.clear();

    // Error Checking
    if (inputs.size() != numInputs)
    {
#if _DEBUG
        printf("NeuralNet - Guess: Invalid Input Vector\n");
#endif
        return outputVector;
    }

    // Translate input vector to input matrix
    inputValues.clear();
    for (int i = 0; i < numInputs; ++i)
    {
        inputValues.setElement(i, 0, inputs[i]);
    }

    // Perform Hidden Layer Multiplication
    hiddenOutput = inputWeights.multiply(inputValues);

    // Add Bias
    hiddenOutput.add(hiddenBias);

    // Apply Activation Function
    hiddenOutput.applyFunction(actFunct);

    // Perform Output Layer Multiplication
    outputValues = outputWeights.multiply(hiddenOutput);

    // Add Bias
    outputValues.add(outputBias);

    // Apply Activation Function
    outputValues.applyFunction(actFunct);

    // Convert Output to Vector
    for (int i = 0; i < numOutputs; ++i)
    {
        outputVector.push_back(outputValues.getElement(i, 0));
    }

    return outputVector;
}

// Train the neural net based on an input vector and a target vector
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::train(const std::vector<double_t>& inputs, const std::vector<double_t>& targets)
{
    // Error Checking
    if (targets.size() != numOutputs)
    {
#if _DEBUG
        printf("NeuralNet - Train: Invalid Target Vector\n");
#endif
        return;
    }

    // Convert Targets to Matrix
    targetMatrix.clear();
    for (int i = 0; i < numOutputs; ++i)
    {
        targetMatrix.setElement(i, 0, targets[i]);
    }
   
    // Make a guess on the inputs
    // guess function updates outputVector and outputValues(matrix)
    guess(inputs); // outputVector = guess(inputs);

    // Calculate the Error at the outputs
    outputError = outputValues.copy();
    outputError.sub(targetMatrix);

    // Calculate Hidden Errors - transposed weights times the error
    hiddenError = (outputWeights.transpose()).multiply(outputError);

    print();
    inputValues.print();
    outputValues.print();
    outputError.print();
    hiddenError.print();
}

// Print the weights and Bias for the entire network
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::print()
{
    printf("Input Weights:\n");
    inputWeights.print();

    printf("Input Bias:\n");
    hiddenBias.print();

    printf("Output Weights:\n");
    outputWeights.print();

    printf("Output Bias:\n");
    outputBias.print();
}
#endif

