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
        // return (input / (1 + abs(input)));

        // Actual Sigmoid
        return ((1.0) / (1.0 + std::exp(-input)));
    }

    double_t sigmoidDerivative(double_t input)
    {
        //double_t sigInput = sigmoid(input);
        //return (sigInput * (1.0 - sigInput));

        // because the sigmoid funciton is already applied to all value matricies
        // The sigmoid does not need to also be applied in the derivative
        return (input * (1.0 - input));
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

    // Randomize the Weights
    void randomize(double_t min, double_t max);

    // Generate an output array based on an input array
    void guess(const double_t (&inputs)[numInputs], double_t (&outputs)[numOutputs]);

    // Train the Neural net based on an input array and an expected answer array
    void train(const double_t(&inputs)[numInputs], const double_t(&answer)[numOutputs]);

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

    /////////////////////////////
    // Feed Fordward Matricies //
    /////////////////////////////
    Matrix<numInputs, 1> inputValues;

    Matrix<numHidden, numInputs> inputWeights;
    Matrix<numHidden, 1> inputBias;

    Matrix<numHidden, 1> hiddenValues;

    Matrix<numOutputs, numHidden> hiddenWeights;
    Matrix<numOutputs, 1> hiddenBias;

    Matrix<numOutputs, 1> outputValues;

    ////////////////////////////////
    // Back Propagation Matricies //
    ////////////////////////////////
    Matrix<numOutputs, 1> answerValues;

    // Error Calculation
    double_t outputArray[numOutputs];
    Matrix<numOutputs, 1> outputError;

    Matrix<numHidden, numOutputs> hiddenWeightsTransposed;
    Matrix<numHidden, 1> hiddenError;

    // Gradient Calculation
    Matrix<numOutputs, 1> outputGradient;
    Matrix<1, numHidden> hiddenValuesTransposed;
    Matrix<numOutputs, numHidden> hiddenWeightsAdjustment;

    Matrix<numHidden, 1> hiddenGradient;
    Matrix<1, numInputs> inputValuesTransposed;
    Matrix<numInputs, numHidden> inputWeightsAdjustment;



    /////////////////////////////
    // Feed Fordward Functions //
    /////////////////////////////
    // Calculate Hidden Layer Values based on Input
    void inputToHidden();

    // Calculate Output Values based on Hidden
    void hiddenToOutput();

    ////////////////////////////////
    // Back Propagation Functions //
    ////////////////////////////////
    // Calculate output error based on output and answers
    void calculateOutputError(const double_t(&answer)[numOutputs]);

    // Calculate output Gradient
    void calculateOutputGradient();

    // Calculate the hidden weight adjustments
    void calculateHiddenWeightDelta(const double_t(&answer)[numOutputs]);

    // Calculate hidden error based on output error and hidden weights
    void calculateHiddenError();

    // Calculate hidden gradient
    void calculateHiddenGradient();

    // Calculate input weight adjustments
    void calculateInputWeightDelta();


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

    // Initialize Feedforward Matricies
    inputValues.clear();
    inputWeights.clear();
    inputBias.clear();

    hiddenValues.clear();
    hiddenWeights.clear();
    hiddenBias.clear();

    outputValues.clear();

    // Initialize back progagation Matricies
    answerValues.clear();
    
    for (uint16_t i = 0; i < numOutputs; ++i)
    {
        outputArray[i] = 0.0;
    }
    outputError.clear();

    hiddenWeightsTransposed.clear();
    hiddenError.clear();
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

// Randomize the Weights
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::randomize(double_t min, double_t max)
{
    inputWeights.randomize(rng, min, max);
    inputBias.randomize(rng, min, max);

    hiddenWeights.randomize(rng, min, max);
    hiddenBias.randomize(rng, min, max);
}

// Generate an output array based on an input array
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::guess(const double_t(&inputs)[numInputs], double_t(&outputs)[numOutputs])
{
    // Reset all intermediate Values
    inputValues.clear();
    hiddenValues.clear();

    // Populate Inputs
    inputValues.fill(inputs);

    // Feed Inputs to Hidden Layer
    inputToHidden();

    // Feed Hidden to Outputs
    hiddenToOutput();

    // Populate output array
    outputValues.toArray(outputs);

}

// Train the Neural net based on an input array and an expected answer array
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::train(const double_t(&inputs)[numInputs], const double_t(&answer)[numOutputs])
{
    // Feed Inputs forward through the Neural Net
    guess(inputs, outputArray);

    // Calculate and Apply Hidden Weight Adjustment - Based on expected output
    calculateHiddenWeightDelta(answer);

    // Calculate and Apply Input Weight Adjustment - Based on hidden layer error
    calculateInputWeightDelta();


}

/////////////////////////////
// Feed Fordward Functions //
/////////////////////////////
// Calculate Hidden Layer Values based on Input
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::inputToHidden()
{
    // Multiply Input Values by Input Weights
    hiddenValues = inputWeights.multiply(inputValues);

    // Add Input Bias
    hiddenValues.add(inputBias);

    // Apply activation funciton
    hiddenValues.applyFunction(actFunct);
}

// Calculate Output Values based on Hidden
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::hiddenToOutput()
{
    // Multiply Hidden values by hidden weights
    outputValues = hiddenWeights.multiply(hiddenValues);

    // Add hidden bias
    outputValues.add(hiddenBias);

    // Apply activation funciton
    outputValues.applyFunction(actFunct);
}

// Calculate output error based on output and answers
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateOutputError(const double_t(&answer)[numOutputs])
{
    // Error = Answers - Outputs
    outputError.fill(answer);

    outputError.sub(outputValues);
}

// Calculate output Gradient
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateOutputGradient()
{
    outputGradient = outputValues;

    // Output * (1 - Output)
    outputGradient.applyFunction(actFunctDeriv);

    // Error times output derivative
    outputGradient.scale(outputError);

    // Scale gradient by learning rate
    outputGradient.scale(learningRate);
}

// Calculate and Apply the hidden wieght adjustments
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateHiddenWeightDelta(const double_t(&answer)[numOutputs])
{
    // Calculate the output Error
    calculateOutputError(answer);

    // Calculate output Gradients
    calculateOutputGradient();

    // Multiply Gradient by Transposed Hidden Values to get Delta Hidden Weights
    hiddenValuesTransposed = hiddenValues.transpose();
    hiddenWeightsAdjustment = outputGradient.multiply(hiddenValuesTransposed);

    // Apply Hidden Weight Adjustments
    hiddenWeights.add(hiddenWeightsAdjustment);
}

// Calculate hidden error based on output error and hidden weights
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateHiddenError()
{
    // Transpose Hidden Weights
    hiddenWeightsTransposed = hiddenWeights.transpose();

    // Calculate Hidden Error
    hiddenError = hiddenWeightsTransposed.multiply(outputError);
}

// Calculate hidden gradient
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateHiddenGradient()
{
    hiddenGradient = hiddenValues;

    // Hidden * (1 - Hidden)
    hiddenGradient.applyFunction(actFunctDeriv);

    // Error times hidden derivative
    hiddenGradient.scale(hiddenError);

    // Scale gradient by learning rate
    hiddenGradient.scale(learningRate);
}

// Calculate and Apply  input weight adjustments
template <uint16_t numInputs, uint16_t numHidden, uint16_t numOutputs>
inline void NeuralNet<numInputs, numHidden, numOutputs>::calculateInputWeightDelta()
{
    // Calculate the Hidden Error
    calculateHiddenError();

    // Calculate the Hidden Gradients
    calculateHiddenGradient();

    // Multiply Gradient by Transposed Hidden Values to get Delta Hidden Weights
    inputValuesTransposed = inputValues.transpose();
    inputWeightsAdjustment = hiddenGradient.multiply(inputValuesTransposed);

    // Apply Hidden Weight Adjustments
    inputWeights.add(inputWeightsAdjustment);
}
#endif

