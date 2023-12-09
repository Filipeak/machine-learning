#include "XORExample.h"
#include <iostream>

void XORExample::Prepare()
{
	m_Inputs = {
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	};
	m_Outputs = {
		{0},
		{1},
		{1},
		{1},
	};

	std::vector<size_t> layers = { 2, 1 };

	m_NN = new NeuralNetwork(layers, NNActivationFunction::Sigmoid);
	m_NN->RandomizeLayers(0.0f, 1.0f);

	m_Grad = new NNData();
	m_Grad->Alloc(layers);
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
			m_NN->Train_Backpropagation(*m_Grad, m_Inputs, m_Outputs);
		}
		else
		{
			m_NN->Train_FiniteDifference(*m_Grad, m_Inputs, m_Outputs, m_DerivativeEps);
		}

		m_NN->Learn(*m_Grad, m_LearningRate);
		m_Grad->Clear();

		m_CurrentIteration++;
	}
	else
	{
		std::cout << " ======= Current Neural Network Data ======= ";
		std::cout << " > Cost: " << m_NN->CalculateCost(m_Inputs, m_Outputs) << std::endl;

		m_NN->Print();

		std::vector<float> params = { 0, 1 };
		std::vector<float> result = m_NN->Forward(params);

		for (size_t i = 0; i < result.size(); i++)
		{
			std::cout << "Result " << i << ": " << result[i] << std::endl;
		}

		std::cout << "==========================================" << std::endl;

		delete m_NN;
		delete m_Grad;

		m_Finished = true;
	}
}

float XORExample::GetCost()
{
	if (!m_Finished)
	{
		return m_NN->CalculateCost(m_Inputs, m_Outputs);
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