#ifdef _WIN32
#include <windows.h>
HANDLE hStdout = GetStdHandle(STD_OUTPUT_HANDLE);
#else

#endif

#include "Matrix.h"
#include "NeuralNet.h"


int main()
{
    // Random Number Generator
    std::mt19937 rng((uint32_t)std::time(0));
    std::uniform_real_distribution<float> uniformDist(-1.0, 1.0);

    std::unique_ptr<NeuralNet<2, 2, 1>> brain = std::make_unique<NeuralNet<2, 2, 1>>(NN::Activations::RELU, 0.01);

    // XOR - Training Set
    std::vector<std::vector<double_t>> xorInputs;
    xorInputs.push_back(std::vector<double_t>{1.0, 0.0});
    xorInputs.push_back(std::vector<double_t>{0.0, 1.0});
    xorInputs.push_back(std::vector<double_t>{1.0, 1.0});
    xorInputs.push_back(std::vector<double_t>{0.0, 0.0});

    std::vector<std::vector<double_t>> xorLabels;
    xorLabels.push_back(std::vector<double_t>{1.0});
    xorLabels.push_back(std::vector<double_t>{1.0});
    xorLabels.push_back(std::vector<double_t>{0.0});
    xorLabels.push_back(std::vector<double_t>{0.0});



    // Batch 
    int batch = 0;
    double_t error = 0;
    double_t largestError = 1;
    while (std::abs(largestError) > 0.05 && batch < 1000)
    {
        // Print stats from last batch
#ifdef _WIN32
        SetConsoleCursorPosition(hStdout, _COORD{ 0, 0 });
#else
        // Move up 5 lines
        printf("\x5b[A");
#endif
        printf("Batch %u\n", ++batch);

        std::vector<double_t> output;
        largestError = 0.0;
        for (int i = 0; i < xorInputs.size(); ++i)
        {
            output = brain->guess(xorInputs[i]);
            printf("Guess: %.2f\n", output[0]);
            printf("Actual: %.2f\n", xorLabels[i][0]);

            error = xorLabels[i][0] - output[0];
            printf("Error: %.2f\n\n", error);

            if (std::abs(error) > std::abs(largestError))
            {
                largestError = error;
            }
        }
        printf("largestError: %.2f\n", largestError);
        //brain->print();
        //system("pause");

        // Train
        for (int count = 0; count < 10000; ++count)
        {
            for (int i = 0; i < xorInputs.size(); ++i)
            {
                brain->train(xorInputs[i], xorLabels[i]);
            }
        }
    }

    if (std::abs(largestError) < 0.05)
    {
        printf("Way to go!! Neural Net was trained!!\n");
    }
    else
    {
        printf("Wow you suck!!! Neural Net is broken!!\n");
        brain->print();
    }
    system("pause");

    // Matrix Testing
    /*
    std::unique_ptr<Matrix<2, 3>> m1 = std::make_unique<Matrix<2, 3>>;
    std::unique_ptr<Matrix<2, 3>> m2 = std::make_unique<Matrix<2, 3>>;
    Matrix<3, 2> m3;
    m1->print(); // Should be 2x3 0s
    m1->add(1);
    m2->add(2);
    m3.add(0.5);
    m1->print(); // Should be 2x3 1s
    m1->multiply(2);
    m1->print(); // Should be 2x3 2s
    m1->add(*m2);
    m1->print(); // Should be 2x3 4s
    Matrix<2, 2> m4 = m1->multiply(m3);
    m3.print(); // Should be 3x2 0.5s
    m4.print(); // Should be 2x2 6s
    m2->setElement(0, 1, 3);
    m2->print(); // Should be 2x3 2 3 2, 2 2 2
    m3 = m2->transpose();
    m3.print(); // Should be 3x2 2 2, 3 2, 2 2
    std::unique_ptr<Matrix<3, 3>> m5 = std::make_unique<Matrix<3, 3>>;
    m5->randomize(-5, 5);
    m5->print(); // Should be 3x3 with random values

    Matrix<2, 1> m6;
    m6.add(2);
    m4.print();
    m6.print();
    Matrix<2, 1> m7 = m4.multiply(m6);
    m7.print();
    */
}