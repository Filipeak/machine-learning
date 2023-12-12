#include "NeuralNetwork.h"
#include "../Utils/Assert.h"
#include <cmath>
#include <algorithm>
#include <iostream>

void NNData::Alloc(const std::vector<size_t>& architecture)
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
	m_Gradient.Alloc(architecture);
	m_BatchesCount = 0;

	SetActivationFunctions(func);
}

void NeuralNetwork::Train_Backpropagation()
{
	m_Gradient.Clear();

	const std::vector<NNBatch> batches = GetBatches();

	const size_t inputs_size = batches.size();
	const size_t last_layer_index = m_Layers.size() - 1;

	std::vector<Matrix> acts;

	for (size_t i = 0; i < m_Layers.size(); i++)
	{
		acts.push_back(Matrix(m_Layers[i].GetRowsSize(), 1));
	}

	for (size_t i = 0; i < inputs_size; i++)
	{
		const std::vector<float> result = Forward(batches[i].inputs);

		for (size_t l = last_layer_index; l > 0; l--)
		{
			for (size_t j = 0; j < m_Layers[l].GetRowsSize(); j++)
			{
				if (l == last_layer_index)
				{
					acts[l].ValueAt(j, 0) = 2 * (result[j] - batches[i].outputs[j]);
				}

				m_Gradient.biases[l - 1].ValueAt(j, 0) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0));

				for (size_t k = 0; k < m_Layers[l - 1].GetRowsSize(); k++)
				{
					m_Gradient.weights[l - 1].ValueAt(j, k) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0)) * m_Layers[l - 1].ValueAt(k, 0);

					acts[l - 1].ValueAt(k, 0) += acts[l].ValueAt(j, 0) * m_ActivationFunctionDerivative(m_LayersRaw[l].ValueAt(j, 0)) * m_Data.weights[l - 1].ValueAt(j, k);
				}

				acts[l].ValueAt(j, 0) = 0;
			}
		}
	}

	for (size_t i = 0; i < m_Gradient.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Gradient.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Gradient.weights[i].GetColumnsSize(); k++)
			{
				m_Gradient.weights[i].ValueAt(j, k) /= inputs_size;
			}
		}
	}

	for (size_t i = 0; i < m_Gradient.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Gradient.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Gradient.biases[i].GetColumnsSize(); k++)
			{
				m_Gradient.biases[i].ValueAt(j, k) /= inputs_size;
			}
		}
	}
}

void NeuralNetwork::Train_FiniteDifference()
{
	m_Gradient.Clear();

	const std::vector<NNBatch> batches = GetBatches();

	const float base_cost = CalculateCost(batches);

	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				float lastValue = m_Data.weights[i].ValueAt(j, k);

				m_Data.weights[i].ValueAt(j, k) += DERIVATIVE_EPS;

				float grad = (CalculateCost(batches) - base_cost) / DERIVATIVE_EPS;

				m_Data.weights[i].ValueAt(j, k) = lastValue;

				m_Gradient.weights[i].ValueAt(j, k) = grad;
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

				m_Data.biases[i].ValueAt(j, k) += DERIVATIVE_EPS;

				float grad = (CalculateCost(batches) - base_cost) / DERIVATIVE_EPS;

				m_Data.biases[i].ValueAt(j, k) = lastValue;

				m_Gradient.biases[i].ValueAt(j, k) = grad;
			}
		}
	}
}

void NeuralNetwork::SetTrainingData(const array_2d_t& inputs, const array_2d_t& outputs)
{
	ASSERT(inputs.size() == outputs.size());

	m_Inputs = inputs;
	m_Outputs = outputs;
}

void NeuralNetwork::SetStochastic(size_t batchesCount)
{
	ASSERT(batchesCount <= m_Inputs.size());

	m_BatchesCount = batchesCount;
}

void NeuralNetwork::Learn(float rate)
{
	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				m_Data.weights[i].ValueAt(j, k) -= rate * m_Gradient.weights[i].ValueAt(j, k);
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].ValueAt(j, k) -= rate * m_Gradient.biases[i].ValueAt(j, k);
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

float NeuralNetwork::CalculateCost(const std::vector<NNBatch>& batches)
{
	float cost = 0.0f, d = 0.0f;
	size_t n = batches.size();

	for (size_t i = 0; i < n; i++)
	{
		const std::vector<float> result = Forward(batches[i].inputs);

		for (size_t j = 0; j < result.size(); j++)
		{
			d = result[j] - batches[i].outputs[j];
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

std::vector<NNBatch> NeuralNetwork::GetBatches() const
{
	std::vector<NNBatch> batches;

	if (m_BatchesCount > 0)
	{
		std::vector<size_t> indexes;

		indexes.reserve(m_Inputs.size());

		for (size_t i = 0; i < m_Inputs.size(); i++)
		{
			indexes.push_back(i);
		}

		std::random_shuffle(indexes.begin(), indexes.end());

		for (size_t i = 0; i < m_BatchesCount; i++)
		{
			batches.push_back({ m_Inputs[indexes[i]], m_Outputs[indexes[i]] });
		}
	}
	else
	{
		for (size_t i = 0; i < m_Inputs.size(); i++)
		{
			batches.push_back({ m_Inputs[i], m_Outputs[i] });
		}
	}

	return batches;
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