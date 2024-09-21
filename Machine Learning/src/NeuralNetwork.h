#pragma once

#include <vector>
#include <string>
#include "NNUtils.h"

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func);
	NeuralNetwork(const std::string& path);

	void Train_Backpropagation();
	void Learn(float rate);
	std::vector<float> Feedforward(const std::vector<float>& params);

	void SetTrainingData(const std::vector<std::vector<float>>& inputs, const std::vector<std::vector<float>>& outputs);
	void SetStochastic(size_t batchesCount);
	void RandomizeLayers(float min, float max);
	float CalculateCost();

	void SaveToFile(const std::string& path) const;

private:
	NNData m_Data;
	NNData m_Gradient;
	NNActivationFunction m_Func;
	std::vector<std::vector<float>> m_Inputs;
	std::vector<std::vector<float>> m_Outputs;
	size_t m_BatchesCount;
	std::vector<Matrix> m_Layers;
	std::vector<Matrix> m_LayersRaw;

	std::vector<NNBatch> GetBatches() const;
};