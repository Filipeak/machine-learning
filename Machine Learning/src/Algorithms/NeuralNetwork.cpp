#include "NeuralNetwork.h"
#include "../Utils/Assert.h"
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>

void NNData::Alloc(const std::vector<size_t> architecture)
{
	arch = architecture;

	for (size_t i = 1; i < architecture.size(); i++)
	{
		weights.push_back(Matrix(architecture[i], architecture[i - 1]));
		biases.push_back(Matrix(architecture[i], 1));
	}
}

void NNData::Clear()
{
	for (size_t i = 0; i < weights.size(); i++)
	{
		for (size_t j = 0; j < weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < weights[i].GetColumnsSize(); k++)
			{
				weights[i].ValueAt(j, k) = 0;
			}
		}
	}

	for (size_t i = 0; i < biases.size(); i++)
	{
		for (size_t j = 0; j < biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < biases[i].GetColumnsSize(); k++)
			{
				biases[i].ValueAt(j, k) = 0;
			}
		}
	}
}

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func)
{
	for (size_t i = 0; i < architecture.size(); i++)
	{
		m_Layers.push_back(Matrix(architecture[i], 1));
		m_LayersRaw.push_back(Matrix(architecture[i], 1));
	}

	m_Data.Alloc(architecture);

	SetActivationFunctions(func);
}

void NeuralNetwork::Train_Backpropagation(NNData& gradient, const array_2d_t& inputs, const array_2d_t& outputs)
{
	ASSERT(inputs.size() == outputs.size());

	const size_t inputs_size = inputs.size();
	const size_t last_layer_index = m_Layers.size() - 1;

	std::vector<Matrix> acts;

	for (size_t i = 0; i < m_Layers.size(); i++)
	{
		acts.push_back(Matrix(m_Layers[i].GetRowsSize(), 1));
	}

	for (size_t i = 0; i < inputs_size; i++)
	{
		const std::vector<float>& result = Forward(inputs[i]);

		for (size_t l = last_layer_index; l > 0; l--)
		{
			for (size_t j = 0; j < m_Layers[l].GetRowsSize(); j++)
			{
				if (l == last_layer_index)
				{
					acts[l].ValueAt(j, 0) = 2 * (result[j] - outputs[i][j]);
				}

				gradient.biases[l - 1].ValueAt(j, 0) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0));

				for (size_t k = 0; k < m_Layers[l - 1].GetRowsSize(); k++)
				{
					gradient.weights[l - 1].ValueAt(j, k) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0)) * m_Layers[l - 1].ValueAt(k, 0);

					acts[l - 1].ValueAt(k, 0) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0)) * m_Data.weights[l - 1].ValueAt(j, k);
				}

				acts[l].ValueAt(j, 0) = 0;
			}
		}
	}

	for (size_t i = 0; i < gradient.weights.size(); i++)
	{
		for (size_t j = 0; j < gradient.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < gradient.weights[i].GetColumnsSize(); k++)
			{
				gradient.weights[i].ValueAt(j, k) /= inputs_size;
			}
		}
	}

	for (size_t i = 0; i < gradient.biases.size(); i++)
	{
		for (size_t j = 0; j < gradient.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < gradient.biases[i].GetColumnsSize(); k++)
			{
				gradient.biases[i].ValueAt(j, k) /= inputs_size;
			}
		}
	}
}

void NeuralNetwork::Train_FiniteDifference(NNData& gradient, const array_2d_t& inputs, const array_2d_t& outputs, float eps)
{
	ASSERT(inputs.size() == outputs.size());

	const float base_cost = CalculateCost(inputs, outputs);

	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				float lastValue = m_Data.weights[i].ValueAt(j, k);

				m_Data.weights[i].ValueAt(j, k) += eps;

				float grad = (CalculateCost(inputs, outputs) - base_cost) / eps;

				m_Data.weights[i].ValueAt(j, k) = lastValue;

				gradient.weights[i].ValueAt(j, k) = grad;
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				float lastValue = m_Data.biases[i].ValueAt(j, k);

				m_Data.biases[i].ValueAt(j, k) += eps;

				float grad = (CalculateCost(inputs, outputs) - base_cost) / eps;

				m_Data.biases[i].ValueAt(j, k) = lastValue;

				gradient.biases[i].ValueAt(j, k) = grad;
			}
		}
	}
}

void NeuralNetwork::Learn(NNData& gradient, float rate)
{
	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				m_Data.weights[i].ValueAt(j, k) -= rate * gradient.weights[i].ValueAt(j, k);
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].ValueAt(j, k) -= rate * gradient.biases[i].ValueAt(j, k);
			}
		}
	}
}

std::vector<float> NeuralNetwork::Forward(const std::vector<float>& params)
{
	ASSERT(m_Layers.size() == params.size());

	for (size_t i = 0; i < m_Layers[0].GetRowsSize(); i++)
	{
		m_Layers[0].ValueAt(i, 0) = params[i];
	}

	for (size_t i = 0; i < m_Layers.size() - 1; i++)
	{
		m_LayersRaw[i + 1] = m_Data.weights[i] * m_Layers[i] + m_Data.biases[i];

		Matrix result(m_LayersRaw[i + 1].GetRowsSize(), 1);

		for (size_t j = 0; j < m_LayersRaw[i + 1].GetRowsSize(); j++)
		{
			result.ValueAt(j, 0) = m_ActivationFunction(m_LayersRaw[i + 1].ValueAt(j, 0));
		}

		m_Layers[i + 1] = result;
	}

	std::vector<float> res;
	Matrix& lastLayer = m_Layers[m_Layers.size() - 1];

	for (size_t i = 0; i < lastLayer.GetRowsSize(); i++)
	{
		res.push_back(lastLayer.ValueAt(i, 0));
	}

	return res;
}

float NeuralNetwork::CalculateCost(const array_2d_t& inputs, const array_2d_t& outputs)
{
	ASSERT(inputs.size() == outputs.size());

	float cost = 0.0f, d = 0.0f;
	size_t n = inputs.size();

	for (size_t i = 0; i < n; i++)
	{
		const std::vector<float>& result = Forward(inputs[i]);

		for (size_t j = 0; j < result.size(); j++)
		{
			d = result[j] - outputs[i][j];
			cost += d * d;
		}
	}

	return cost / n;
}

void NeuralNetwork::RandomizeLayers(float min, float max)
{
	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				m_Data.weights[i].ValueAt(j, k) = (max - min) * RandomFloat01() + min;
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].ValueAt(j, k) = (max - min) * RandomFloat01() + min;
			}
		}
	}
}

const NNData& NeuralNetwork::GetData() const
{
	return m_Data;
}

void NeuralNetwork::Print()
{
	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		std::cout << " > Weights " << i << std::endl;
		std::cout << m_Data.weights[i] << std::endl;

		std::cout << " > Biases " << i << std::endl;
		std::cout << m_Data.biases[i] << std::endl;
	}
}

void NeuralNetwork::SetActivationFunctions(NNActivationFunction func)
{
	m_Func = func;

	switch (func)
	{
	case NNActivationFunction::Sigmoid:
		m_ActivationFunction = Sigmoid;
		m_ActivationFunctionDerivative = Sigmoid_Derivative;

		break;
	case NNActivationFunction::ReLU:
		m_ActivationFunction = ReLU;
		m_ActivationFunctionDerivative = ReLU_Derivative;

		break;
	case NNActivationFunction::Tanh:
		m_ActivationFunction = Tanh;
		m_ActivationFunctionDerivative = Tanh_Derivative;

		break;
	default:
		break;
	}
}

float NeuralNetwork::RandomFloat01()
{
	return (float)rand() / (float)RAND_MAX;
}

float NeuralNetwork::Sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float NeuralNetwork::Sigmoid_Derivative(float x)
{
	return Sigmoid(x) * (1.0f - Sigmoid(x));
}

float NeuralNetwork::ReLU(float x)
{
	return x > 0 ? x : 0;
}

float NeuralNetwork::ReLU_Derivative(float x)
{
	return x > 0.0f ? 1.0f : 0.0f;
}

float NeuralNetwork::Tanh(float x)
{
	return tanhf(x);
}

float NeuralNetwork::Tanh_Derivative(float x)
{
	float t = Tanh(x);

	return 1.0f - t * t;
}