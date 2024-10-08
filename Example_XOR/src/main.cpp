﻿#include <iostream>
#include <NeuralNetwork.h>

int main()
{
	const std::vector<std::vector<float>> inputs = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	const std::vector<std::vector<float>> outputs = {
		{0},
		{1},
		{1},
		{1},
	};

	NeuralNetwork nn({ 2, 1 }, NNActivationFunction::Sigmoid);
	nn.RandomizeLayers(-2.0f, 2.0f);
	nn.SetTrainingData(inputs, outputs);

	for (size_t i = 0; i < 25000; i++)
	{
		nn.Train_Backpropagation();
		nn.Learn(0.1f);
	}

	std::cout << "Cost: " << nn.CalculateCost() << std::endl;

	std::vector<float> params = { 0, 1 };
	std::vector<float> result = nn.Feedforward(params);

	std::cout << "0 ^ 1 = " << result[0] << std::endl;

	std::cout << "==========================================" << std::endl;
}