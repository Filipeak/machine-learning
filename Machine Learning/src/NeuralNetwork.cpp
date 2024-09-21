#include "NeuralNetwork.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>

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

				m_Data.weights[weightIndex].GetValueAtRef(weightRow, i) = x;
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

				m_Data.biases[biasIndex].GetValueAtRef(i, 0) = x;
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
		const std::vector<float> result = Feedforward(batches[i].inputs);

		for (size_t l = last_layer_index; l > 0; l--)
		{
			for (size_t j = 0; j < m_Layers[l].GetRowsSize(); j++)
			{
				if (l == last_layer_index)
				{
					acts[l].GetValueAtRef(j, 0) = 2 * (result[j] - batches[i].outputs[j]);
				}

				m_Gradient.biases[l - 1].GetValueAtRef(j, 0) += acts[l].GetValueAtRef(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].GetValueAtRef(j, 0));

				for (size_t k = 0; k < m_Layers[l - 1].GetRowsSize(); k++)
				{
					m_Gradient.weights[l - 1].GetValueAtRef(j, k) += acts[l].GetValueAtRef(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].GetValueAtRef(j, 0)) * m_Layers[l - 1].GetValueAtRef(k, 0);

					acts[l - 1].GetValueAtRef(k, 0) += acts[l].GetValueAtRef(j, 0) * NNActivationFuncs::CallFuncDerivative(m_Func, m_LayersRaw[l].GetValueAtRef(j, 0)) * m_Data.weights[l - 1].GetValueAtRef(j, k);
				}

				acts[l].GetValueAtRef(j, 0) = 0;
			}
		}
	}

	for (size_t i = 0; i < m_Gradient.weights.size(); i++)
	{
		for (size_t j = 0; j < m_Gradient.weights[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Gradient.weights[i].GetColumnsSize(); k++)
			{
				m_Gradient.weights[i].GetValueAtRef(j, k) /= inputs_size;
			}
		}
	}

	for (size_t i = 0; i < m_Gradient.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Gradient.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Gradient.biases[i].GetColumnsSize(); k++)
			{
				m_Gradient.biases[i].GetValueAtRef(j, k) /= inputs_size;
			}
		}
	}
}

void NeuralNetwork::SetTrainingData(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs)
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
				m_Data.weights[i].GetValueAtRef(j, k) -= rate * m_Gradient.weights[i].GetValueAtRef(j, k);
				m_Gradient.weights[i].GetValueAtRef(j, k) = 0;
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].GetValueAtRef(j, k) -= rate * m_Gradient.biases[i].GetValueAtRef(j, k);
				m_Gradient.biases[i].GetValueAtRef(j, k) = 0;
			}
		}
	}
}

float NeuralNetwork::CalculateCost()
{
	const std::vector<NNBatch> batches = GetBatches();

	float cost = 0.0f, d = 0.0f;
	size_t n = batches.size();

	for (size_t i = 0; i < n; i++)
	{
		const std::vector<float> result = Feedforward(batches[i].inputs);

		for (size_t j = 0; j < result.size(); j++)
		{
			d = result[j] - batches[i].outputs[j];
			cost += d * d;
		}
	}

	return cost / n;
}

std::vector<float> NeuralNetwork::Feedforward(const std::vector<float>& params)
{
	NN_ASSERT(m_Layers.size() == params.size());

	for (size_t i = 0; i < m_Layers[0].GetRowsSize(); i++)
	{
		m_Layers[0].GetValueAtRef(i, 0) = params[i];
	}

	for (size_t i = 0; i < m_Layers.size() - 1; i++)
	{
		m_LayersRaw[i + 1] = m_Data.weights[i] * m_Layers[i] + m_Data.biases[i];

		for (size_t j = 0; j < m_LayersRaw[i + 1].GetRowsSize(); j++)
		{
			m_Layers[i + 1].GetValueAtRef(j, 0) = NNActivationFuncs::CallFunc(m_Func, m_LayersRaw[i + 1].GetValueAtRef(j, 0));
		}
	}

	std::vector<float> res;
	const Matrix& lastLayer = m_Layers[m_Layers.size() - 1];

	for (size_t i = 0; i < lastLayer.GetRowsSize(); i++)
	{
		res.push_back(lastLayer.GetValueAt(i, 0));
	}

	return res;
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
				m_Data.weights[i].GetValueAtRef(j, k) = distr(gen);
			}
		}
	}

	for (size_t i = 0; i < m_Data.biases.size(); i++)
	{
		for (size_t j = 0; j < m_Data.biases[i].GetRowsSize(); j++)
		{
			for (size_t k = 0; k < m_Data.biases[i].GetColumnsSize(); k++)
			{
				m_Data.biases[i].GetValueAtRef(j, k) = distr(gen);
			}
		}
	}
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