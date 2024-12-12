#pragma once

#include "Matrix.h"
#include "NeuralNet.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

const uint16_t IMG_WIDTH = 28;
const uint16_t IMG_LEN = IMG_WIDTH * IMG_WIDTH;

struct minstImage
{
    uint16_t label;

    double_t image[IMG_LEN];

    minstImage(uint8_t lab) : label(lab), image{0.0} { ; }
};

uint16_t MAX_IMAGES = 0xFFFF;

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
                std::cout << img->label;
            }
            else
            {
                std::cout << ' ';
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
                img->image[j] = tmp / 255.0;
            }
        }
        //drawImage(img);
    }

}

void minstMain()
{
    std::mt19937 rng((uint32_t)std::time(0));

    importData();

    std::cout << "Data Imported" << std::endl;

    const uint16_t numHidden = 300;
    const uint16_t numOutput = 10;

    NeuralNet<IMG_LEN, numHidden, numOutput>* brain = new NeuralNet<IMG_LEN, numHidden, numOutput>(rng,    // Random Number Generator
        NN::Activations::SIGMOID, // Activation Function
        0.01); // Learning Rate

    std::cout << "Brain Created" << std::endl;

    double_t output[numOutput] = { 0.0};

    brain->guess(trainingSet[0].image, output);

    for (int i = 0; i < numOutput; ++i)
    {
        std::cout << i << " : " << output[i] << std::endl;
    }
    drawImage(&trainingSet[0]);

    std::uniform_real_distribution<float> uniformDist(0, numTraining);

    uint32_t numTraining = 100000;
    uint16_t batchSize = 1000;
    uint32_t numCycles = numTraining / batchSize;

    for (uint32_t i = 0; i < numCycles; ++i)
    {
        for (int j = 0; j < batchSize; ++j)
        {
            for (int k = 0; k < numOutput; ++k)
            {
                output[k] = 0.0;
            }

            int idx = uniformDist(rng);

            output[trainingSet[idx].label] = 1.0;
            brain->train(trainingSet[0].image, output);
        }

        double_t threshold = 0.05;

        for (int k = 0; k < numOutput; ++k)
        {
            output[k] = 0.0;
        }

        int idx = uniformDist(rng);

        output[trainingSet[idx].label] = 1.0;

        double_t error = brain->test(trainingSet[idx].image, output);

        if (error < threshold)
        {
            printf("Confident within %.03f with error %f after %d cycles", threshold, error, i);
            break;
        }

        std::cout << "Trained: " << i * batchSize << " - Error: " << error << std::endl;

    }


    brain->guess(trainingSet[0].image, output);

    for (int i = 0; i < numOutput; ++i)
    {
        std::cout << i << " : " << output[i] << std::endl;
    }
    drawImage(&trainingSet[0]);

    delete brain;
}