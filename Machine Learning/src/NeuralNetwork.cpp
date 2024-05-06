#include "NeuralNetwork.h"
#include "NNAssert.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

#define DERIVATIVE_EPS 0.001f;

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

NeuralNetwork::NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func) : m_Func(func), m_BatchesCount(0)
{
	for (size_t i = 0; i < architecture.size(); i++)
	{
		m_Layers.push_back(Matrix(architecture[i], 1));
		m_LayersRaw.push_back(Matrix(architecture[i], 1));
	}

	m_Data.Alloc(architecture);
	m_Gradient.Alloc(architecture);
}

NeuralNetwork::NeuralNetwork(const std::string& path) : m_Func(NNActivationFunction::Sigmoid), m_BatchesCount(0)
{
	std::fstream file(path);
	std::string line;

	size_t weightIndex = 0, weightRow = 0, biasIndex = 0;

	bool parsingWeights = false;
	bool parsingBiases = false;

	while (std::getline(file, line))
	{
		if (!parsingWeights && !parsingBiases)
		{
			std::vector<size_t> arch;
			bool funcSet = false;

			int tmp = 0;

			for (size_t i = 0; i < line.size(); i++)
			{
				if (line[i] == ' ')
				{
					if (!funcSet)
					{
						m_Func = (NNActivationFunction)tmp;

						funcSet = true;
					}
					else
					{
						arch.push_back((size_t)tmp);
					}

					tmp = 0;
				}
				else
				{
					tmp = tmp * 10 + (int)(line[i] - '0');
				}
			}

			for (size_t i = 0; i < arch.size(); i++)
			{
				m_Layers.push_back(Matrix(arch[i], 1));
				m_LayersRaw.push_back(Matrix(arch[i], 1));
			}

			m_Data.Alloc(arch);
			m_Gradient.Alloc(arch);

			parsingWeights = true;
		}
		else if (parsingWeights)
		{
			std::istringstream ss(line);

			for (size_t i = 0; i < m_Data.weights[weightIndex].GetColumnsSize(); i++)
			{
				float x;
				ss >> x;

				m_Data.weights[weightIndex].ValueAt(weightRow, i) = x;
			}

			weightRow++;

			if (weightRow == m_Data.weights[weightIndex].GetRowsSize())
			{
				weightRow = 0;
				weightIndex++;

				if (weightIndex == m_Data.weights.size())
				{
					parsingWeights = false;
					parsingBiases = true;
				}
			}
		}
		else if (parsingBiases)
		{
			std::istringstream ss(line);

			for (size_t i = 0; i < m_Data.biases[biasIndex].GetRowsSize(); i++)
			{
				float x;
				ss >> x;

				m_Data.biases[biasIndex].ValueAt(i, 0) = x;
			}

			biasIndex++;

			if (biasIndex == m_Data.biases.size())
			{
				parsingBiases = false;
			}
		}
	}

	file.close();
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

				m_Gradient.biases[l - 1].ValueAt(j, 0) += acts[l].ValueAt(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].ValueAt(j, 0));

				for (size_t k = 0; k < m_Layers[l - 1].GetRowsSize(); k++)
				{
					m_Gradient.weights[l - 1].ValueAt(j, k) += acts[l].ValueAt(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].ValueAt(j, 0)) * m_Layers[l - 1].ValueAt(k, 0);

					acts[l - 1].ValueAt(k, 0) += acts[l].ValueAt(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].ValueAt(j, 0)) * m_Data.weights[l - 1].ValueAt(j, k);
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
	NN_ASSERT(inputs.size() == outputs.size());

	m_Inputs = inputs;
	m_Outputs = outputs;
}

void NeuralNetwork::SetStochastic(size_t batchesCount)
{
	NN_ASSERT(batchesCount <= m_Inputs.size());

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
	NN_ASSERT(m_Layers.size() == params.size());

	for (size_t i = 0; i < m_Layers[0].GetRowsSize(); i++)
	{
		m_Layers[0].ValueAt(i, 0) = params[i];
	}

	for (size_t i = 0; i < m_Layers.size() - 1; i++)
	{
		m_LayersRaw[i + 1] = m_Data.weights[i] * m_Layers[i] + m_Data.biases[i];

		for (size_t j = 0; j < m_LayersRaw[i + 1].GetRowsSize(); j++)
		{
			m_Layers[i + 1].ValueAt(j, 0) = NNActivationFuncs::CallFunc(m_Func, m_LayersRaw[i + 1].ValueAt(j, 0));
		}
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
	std::random_device dev;
	std::default_random_engine gen(dev());
	std::uniform_real_distribution<float> distr(min, max);

	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				m_Data.weights[i].ValueAt(j, k) = distr(gen);
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].ValueAt(j, k) = distr(gen);
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

void NeuralNetwork::SaveToFile(const std::string& path) const
{
	std::ofstream file(path);

	file << (int)m_Func << " ";

	for (size_t i = 0; i < m_Data.arch.size(); i++)
	{
		file << m_Data.arch[i] << " ";
	}

	file << std::endl;

	for (size_t i = 0; i < m_Data.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Data.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.weights[i].GetColumnsSize(); k++)
			{
				file << m_Data.weights[i].GetValueAt(j, k) << " ";
			}

			file << std::endl;
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			file << m_Data.biases[i].GetValueAt(j, 0) << " ";
		}

		if (i < m_Data.biases.size() - 1)
		{
			file << std::endl;
		}
	}

	file.close();

	std::cout << "Successfully saved neural network to file: " << path << std::endl;
}