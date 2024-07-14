#include "Matrix.h"
#include "NeuralNet.h"


int main()
{
    // Random Number Generator
    std::mt19937 rng((uint32_t)std::time(0));
    std::uniform_real_distribution<float> uniformDist(-1.0, 1.0);

    std::unique_ptr<NeuralNet<2, 2, 2>> brain = std::make_unique<NeuralNet<2, 2, 2>>(NN::Activations::SIGMOID, 0.1);
    //brain->print();

    // Random Input for Neural Net Testing
    std::vector<double_t> in = { 1, 0 };
    std::vector<double_t> target = { 1, 0 };
    brain->train(in, target);
    std::vector<double_t> out = brain->guess(in);

    printf("Output:\n");
    for (int i = 0; i < out.size(); ++i)
    {
        printf("%.2f\n", out[i]);
    }

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