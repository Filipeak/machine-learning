/**
* RESOURCES:
*  - 3Blue1Brown Explanation 1: https://www.youtube.com/watch?v=aircAruvnKk
*  - 3Blue1Brown Explanation 2: https://www.youtube.com/watch?v=IHZwWFHWa-w
*  - 3Blue1Brown Explanation 3: https://www.youtube.com/watch?v=Ilg3gGewQ5U
*  - 3Blue1Brown Explanation 4: https://www.youtube.com/watch?v=tIeHLnjs5U8
*  - Example Series: https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw
*
* IDEAS:
*  - Stochastic Gradient Descent
*  - Exporting / Importing
*  - Visualizing (Network, Cost Graph) - SDL
*  - Drawing Example (Visualized) - SDL
*  - Code in C - Arrays, structs, optimization, ...
*/


#include <iostream>

#include "Algorithms/NeuralNetwork.h"

int main()
{
	std::srand((unsigned int)std::time(0));
	
	std::vector<size_t> layers = { 2, 1 };
	array_2d_t inputs = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	array_2d_t outputs = {
		{0},
		{1},
		{1},
		{1},
	};
	std::vector<float> params = { 0, 1 };
	float learningRate = 0.1f;
	float derivativeEps = 0.1f;
	size_t iterations = 10000;

	NeuralNetwork nn(layers, NNActivationFunction::Sigmoid);

	nn.RandomizeLayers(0.0f, 1.0f);

	NNData grad;
	grad.Alloc(layers);

	for (size_t i = 0; i < iterations; i++)
	{
		//nn.Train_FiniteDifference(grad, inputs, outputs, derivativeEps);
		nn.Train_Backpropagation(grad, inputs, outputs);
		nn.Learn(grad, learningRate);

		grad.Clear();
	}

	std::cout << "Cost: " << nn.CalculateCost(inputs, outputs) << std::endl;

	nn.Print();

	std::vector<float> result = nn.Forward(params);

	for (size_t i = 0; i < result.size(); i++)
	{
		std::cout << "Result " << i << ": " << result[i] << std::endl;
	}
}