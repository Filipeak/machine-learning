#include "XORExample.h"
#include <iostream>

void XORExample::Prepare()
{
	const array_2d_t inputs = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	const array_2d_t outputs = {
		{0},
		{1},
		{1},
		{1},
	};

	std::vector<size_t> layers = { 2, 1 };

	m_NN = new NeuralNetwork(layers, NNActivationFunction::Sigmoid);
	m_NN->RandomizeLayers(0.0f, 1.0f);
	m_NN->SetTrainingData(inputs, outputs);
}

void XORExample::RunIteration()
{
	if (m_Finished)
	{
		return;
	}

	if (m_CurrentIteration < m_MaxIterations)
	{
		if (m_Backpropagation)
		{
			m_NN->Train_Backpropagation();
		}
		else
		{
			m_NN->Train_FiniteDifference();
		}

		m_NN->Learn(m_LearningRate);

		m_CurrentIteration++;
	}
	else
	{
		std::cout << " ======= Neural Network Data ======= " << std::endl;

		m_NN->Print();

		std::cout << "Cost: " << m_NN->CalculateCost(m_NN->GetBatches()) << std::endl;

		std::vector<float> params = { 0, 1 };
		std::vector<float> result = m_NN->Forward(params);

		std::cout << "0 ^ 1 = " << result[0] << std::endl;

		std::cout << "==========================================" << std::endl;

		delete m_NN;

		m_Finished = true;
	}
}

float XORExample::GetCost()
{
	if (!m_Finished)
	{
		return m_NN->CalculateCost(m_NN->GetBatches());
	}
	else
	{
		return 0;
	}
}

const NNData& XORExample::GetData() const
{
	return m_NN->GetData();
}