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

};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
class NeuralNet
{
public:
    
    NeuralNet(NN::Activations activation = NN::Activations::SIGMOID, double_t learningRate = 0.001);

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
    double_t(*actFunctDeriv)(double_t);

    // Learning Rate
    double_t learningRate;

    ///////////////////////////////
    //  Feed Forward Matricies   //
    ///////////////////////////////
    // Input Matricies 
    Matrix<numInputs, 1> inputValues;

    // Hidden Matricies
    Matrix<numHidden, numInputs> hiddenWeights;
    Matrix<numHidden, 1> hiddenBias;


    // Output Matricies
    Matrix<numHidden, 1> hiddenValues;
    Matrix<numOutputs, numHidden> outputWeights;
    Matrix<numOutputs, 1> outputBias;

    Matrix<numOutputs, 1> outputValues;
    std::vector<double_t> outputVector;
    
    ///////////////////////////////
    // Backpropagation Matricies //
    ///////////////////////////////
    // Input Matricies

    // Hidden Matricies
    Matrix<numHidden, numInputs> deltaHiddenWeights;
    Matrix<numHidden, 1> hiddenError;
    Matrix<numHidden, 1> hiddenValuesDerivative;

    // Output Matricies
    Matrix<numOutputs, 1> targetMatrix;
    Matrix<numOutputs, numHidden> deltaOutputWeights;
    Matrix<numOutputs, 1> outputError;
    Matrix<numOutputs, 1> outputValuesDerrivative;
};

template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline NeuralNet<numInputs, numHidden, numOutputs>::NeuralNet(NN::Activations activation, double_t learningRate)
    : rng((uint32_t)std::time(0)),
      uniformDist(-1.0, 1.0),
      activationFunciton(activation),
      actFunct(0),
      learningRate(learningRate),
      // Feedforward Matricies
      inputValues(),
      hiddenWeights(),
      hiddenBias(),
      hiddenValues(),
      outputWeights(),
      outputBias(),
      outputValues(),
      outputVector(numOutputs),
      // Backpropagation Matricies
      deltaHiddenWeights(),
      hiddenError(),
      hiddenValuesDerivative(),
      targetMatrix(),
      deltaOutputWeights(),
      outputError(),
      outputValuesDerrivative()
{
    hiddenWeights.randomize(rng, -1.0, 1.0);
    hiddenBias.randomize(rng, -1.0, 1.0);

    outputWeights.randomize(rng, -1.0, 1.0);
    outputBias.randomize(rng, -1.0, 1.0);

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
        for (int i = 0; i < numOutputs; ++i)
        {
            outputVector.push_back(0.0);
        }
        return outputVector;
    }

    // Translate input vector to input matrix
    inputValues.clear();
    for (int i = 0; i < numInputs; ++i)
    {
        inputValues.setElement(i, 0, inputs[i]);
    }

    // Perform Hidden Layer Multiplication
    hiddenValues = hiddenWeights.multiply(inputValues);

    // Add Bias
    hiddenValues.add(hiddenBias);

    // Apply Activation Function
    hiddenValues.applyFunction(actFunct);

    // Perform Output Layer Multiplication
    outputValues = outputWeights.multiply(hiddenValues);

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
    // guess function updates outputValues
    guess(inputs);

    // Catch run-away training
    if (std::abs(hiddenWeights.getElement(0, 0)) > 1000 ||
        std::abs(hiddenBias.getElement(0, 0)) > 1000 ||
        std::abs(outputWeights.getElement(0, 0)) > 1000 ||
        std::abs(outputBias.getElement(0, 0)) > 1000)
    {
        //system("pause");
    }

    // Calculate the Error at the outputs
    outputError = targetMatrix;
    outputError.sub(outputValues);
    //outputError = outputValues;
    //outputError.sub(targetMatrix);
    // Mean Squared Error
    outputError.applyFunction(NN::square);

    // Calculate Hidden Errors - transposed weights times the output error
    hiddenError = (outputWeights.transpose()).multiply(outputError);

    // Calculate Output weight adjustment matrix
    // deltaWeights = learningRate * OutputError * (OutputValues * (1 - OutputValues)) * HiddenValues(transposed)

    //(OutputValues * (1 - OutputValues))
    outputValuesDerrivative = outputValues;
    outputValuesDerrivative.applyFunction(actFunctDeriv);

    // OutputError * (OutputValues * (1 - OutputValues)) - Element-wise multiplication
    //outputError.multiply(outputValuesDerrivative);
    outputValuesDerrivative.multiply(outputError);

    // learningRate * OutputError * (OutputValues * (1 - OutputValues))
    outputValuesDerrivative.multiply(learningRate);

    // Output Bias Adjustment
    // deltaBias = learningRate * OutputError * (OutputValues * (1 - OutputValues))
    // Apply Hidden Bias Corrections
    outputBias.add(outputValuesDerrivative);

    // OutputError * (OutputValues * (1 - OutputValues)) * HiddenValues(transposed)
    deltaOutputWeights = outputValuesDerrivative.multiply(hiddenValues.transpose());

    // Apply Output Corrections
    outputWeights.add(deltaOutputWeights);



    // Calculate Hidden weight adjustment matrix
    // deltaWeights = learningRate * HiddenError * (HiddenValues * (1 - HiddenValues)) * InputValues(transposed)

    // (HiddenValues * (1 - HiddenValues))
    hiddenValuesDerivative = hiddenValues;
    hiddenValuesDerivative.applyFunction(actFunctDeriv);

    // HiddenError * (HiddenValues * (1 - HiddenValues)) - Element-wise multiplications
    //hiddenError.multiply(hiddenValuesDerivative);
    hiddenValuesDerivative.multiply(hiddenError);

    // learningRate * HiddenError * (HiddenValues * (1 - HiddenValues))
    hiddenValuesDerivative.multiply(learningRate);

    // Hidden Bias Adjustment
    // deltaBias = learningRate * HiddenError * (HiddenValues * (1 - HiddenValues))
    // Apply Hidden Bias Corrections
    hiddenBias.add(hiddenValuesDerivative);

    // learningRate * HiddenError * (HiddenValues * (1 - HiddenValues)) * InputValues(transposed)
    deltaHiddenWeights = hiddenValuesDerivative.multiply(inputValues.transpose());

    // Apply Hidden Weight Corrections
    hiddenWeights.add(deltaHiddenWeights);
}

// Print the weights and Bias for the entire network
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::print()
{
    printf("Input Weights:\n");
    hiddenWeights.print();

    printf("Input Bias:\n");
    hiddenBias.print();

    printf("Output Weights:\n");
    outputWeights.print();

    printf("Output Bias:\n");
    outputBias.print();
}
#endif

