#pragma once

#include "Matrix.h"
#include "NeuralNet.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

const uint16_t IMG_WIDTH = 28;
const uint16_t IMG_LEN = IMG_WIDTH * IMG_WIDTH;

const uint16_t numOutput = 10;
const uint16_t numHidden = 100;

std::mt19937 mnistRng((uint32_t)std::time(0));
NeuralNet<IMG_LEN, numHidden, numOutput>* brain = new NeuralNet<IMG_LEN, numHidden, numOutput>(mnistRng,    // Random Number Generator
    NN::Activations::SIGMOID, // Activation Function
    0.001); // Learning Rate


struct minstImage
{
    uint16_t label;

    double_t image[IMG_LEN];

    minstImage(uint8_t lab) : label(lab), image{0.0} { ; }
};

uint16_t MAX_IMAGES = 0xFFFF;

// Only test and train for a subset of digits
uint16_t TESTING_MASK[numOutput] = { 1,  // 0
                                     0,  // 1
                                     0,  // 2
                                     0,  // 3
                                     0,  // 4
                                     1,  // 5
                                     0,  // 6
                                     0,  // 7
                                     0,  // 8
                                     0 };// 9

uint16_t numTraining = 0;
std::vector<minstImage> trainingSet;

uint16_t numTest = 0;
std::vector<minstImage> testSet;

void drawImage(minstImage* img)
{
    for (int i = 0; i < IMG_WIDTH; ++i)
    {
        for (int j = 0; j < IMG_WIDTH; ++j)
        {
            uint16_t idx = i * IMG_WIDTH + j;
            if (img->image[idx] > 0.0)
            {
                std::cout << img->label << ' ';
            }
            else
            {
                std::cout << ' ' << ' ';
            }
        }
        std::cout << std::endl;

    }
}

void importData()
{
    // Get Training Labels
    std::ifstream fin("C:\\Users\\edwar\\Documents\\_Fun\\Code\\NeuralNet\\minstData\\train-labels.idx1-ubyte", std::ios::binary);

    if (!fin.is_open())
    {
#pragma warning(suppress : 4996)
        std::cout << "Error: " << strerror(errno);
        return;
    }

    char tmp = 0;
    
    // Read MAGIC code
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read number of images
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read all Labels
    while (fin)
    {
        fin.read(&tmp, 1);

        // Ensure read completed successfully
        if (fin)
        {
            if (++numTraining < MAX_IMAGES)
            {
                trainingSet.push_back(minstImage(tmp));
            }
        }
    }

    fin.close();

    // Get Training Images
    fin.open("C:\\Users\\edwar\\Documents\\_Fun\\Code\\NeuralNet\\minstData\\train-images.idx3-ubyte", std::ios::binary);

    if (!fin.is_open())
    {
#pragma warning(suppress : 4996)
        std::cout << "Error: " << strerror(errno);
        return;
    }

    // Read number of rows
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // read number of columns
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read MAGIC code
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read number of images
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read all Images
    for (int i = 0; i < numTraining; ++i)
    {
        minstImage* img = &trainingSet[i];

        for (int j = 0; j < IMG_LEN; ++j)
        {
            fin.read(&tmp, 1);

            // Ensure read completed successfully
            if (fin)
            {
                img->image[j] = (uint8_t)tmp / 255.0;
            }
        }
        //drawImage(img);
    }

    fin.close();

    // Get Testing Labels
    fin.open("C:\\Users\\edwar\\Documents\\_Fun\\Code\\NeuralNet\\minstData\\t10k-labels.idx1-ubyte", std::ios::binary);

    if (!fin.is_open())
    {
#pragma warning(suppress : 4996)
        std::cout << "Error: " << strerror(errno);
        return;
    }

    // Read MAGIC code
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read number of images
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read all Labels
    while (fin)
    {
        fin.read(&tmp, 1);

        // Ensure read completed successfully
        if (fin)
        {
            if (++numTest < MAX_IMAGES)
            {
                testSet.push_back(minstImage(tmp));
            }
        }
    }

    fin.close();

    // Get Testing Images
    fin.open("C:\\Users\\edwar\\Documents\\_Fun\\Code\\NeuralNet\\minstData\\t10k-images.idx3-ubyte", std::ios::binary);

    if (!fin.is_open())
    {
#pragma warning(suppress : 4996)
        std::cout << "Error: " << strerror(errno);
        return;
    }

    // Read number of rows
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // read number of columns
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read MAGIC code
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read number of images
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);
    fin.read(&tmp, 1);

    // Read all Images
    for (int i = 0; i < numTest; ++i)
    {
        minstImage* img = &testSet[i];

        for (int j = 0; j < IMG_LEN; ++j)
        {
            fin.read(&tmp, 1);

            // Ensure read completed successfully
            if (fin)
            {
                img->image[j] = (uint8_t)tmp / 255.0;
            }
        }
        //drawImage(img);
    }

}

template <typename T>
int getHighestIndex(T* arr, uint16_t arrSize)
{
    int index = -1;
    T highest = static_cast<T>(0.0);

    for (int i = 0; i < arrSize; ++i)
    {
        if (arr[i] > highest)
        {
            highest = arr[i];
            index = i;
        }
    }

    return index;
}

uint32_t numTrained[10] = { 0 };

void trainEpoch()
{
    double_t answer[numOutput] = { 0.0 };
    for (int k = 0; k < numOutput; ++k)
    {
        answer[k] = 0.0;
    }

    std::uniform_real_distribution<float> uniformDist(0, numTraining);

    uint32_t numImagesTrained = 0;

    while (numImagesTrained < numTraining)
    {
        //int idx = uniformDist(mnistRng);
        // Temp train all images in order
        for (int idx = 0; idx < numTraining; ++idx)
        {
            if (TESTING_MASK[trainingSet[idx].label] == 1 &&
                //true)
                trainingSet[idx].label != getHighestIndex(numTrained, numOutput))
            {
                // Set Correct Answer
                answer[trainingSet[idx].label] = 1.0;

                // Train NN
                brain->train(trainingSet[0].image, answer);
                // Track how many of each digit were trained
                ++numTrained[trainingSet[idx].label];
                ++numImagesTrained;

                // Reset Answer Array
                answer[trainingSet[idx].label] = 0.0;

            }
        }
    }
}

uint32_t numTested[10] = { 0 };

double_t testEpoch()
{
    double_t output[numOutput] = { 0.0 };
    for (int k = 0; k < numOutput; ++k)
    {
        output[k] = 0.0;
    }

    std::uniform_real_distribution<float> uniformDist(0, numTest);

    double_t numImagesTested = 1.0;
    double_t numCorrect = 0.0;

    for(int i = 0; i < numTest; ++i)
    {
        if (TESTING_MASK[testSet[i].label] == 1)
        {
            // Test NN
            brain->guess(testSet[i].image, output);
            // Track how many of each digit were tested
            ++numTested[testSet[i].label];
            ++numImagesTested;

            if (getHighestIndex(output, numOutput) == testSet[i].label)
            {
                ++numCorrect;
            }
        }
    }

    return numCorrect / numImagesTested;
}

void minstMain()
{
    importData();

    std::cout << "Data Imported" << std::endl;


    std::cout << "Brain Created" << std::endl;

    double_t output[numOutput] = { 0.0 };

    brain->guess(trainingSet[0].image, output);

    for (int i = 0; i < numOutput; ++i)
    {
        std::cout << i << " : " << output[i] << std::endl;
    }
    drawImage(&trainingSet[0]);

    std::uniform_real_distribution<float> uniformDist(0, numTraining);

    uint16_t numToTrain = 5;
    uint16_t numEpochs = 0;

    do
    {
        double_t percentCorrect = testEpoch();

        std::cout << "Trained " << numEpochs++ << " Epochs - Accuracy: " << percentCorrect * 100 << "%" << std::endl;

        trainEpoch();

    } while (numEpochs < numToTrain);

    brain->guess(trainingSet[0].image, output);
    uint16_t guess = getHighestIndex(output, numOutput);

    for (int i = 0; i < numOutput; ++i)
    {
        if (i == guess)
        {
            std::cout << i << " (" << numTrained[i] << ")" << " ! " << output[i] << std::endl;
        }
        else
        {
            std::cout << i << " (" << numTrained[i] << ")" << " : " << output[i] << std::endl;
        }
    }
    drawImage(&trainingSet[0]);

    delete brain;
}