#pragma once

#include <vector>
#include <string>
#include "Matrix.h"
#include "NNActivationFuncs.h"

typedef std::vector<std::vector<float>> array_2d_t;

struct NNData
{
	std::vector<size_t> arch;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;

	void Alloc(const std::vector<size_t>& architecture);
	void Clear();
};

struct NNBatch
{
	std::vector<float> inputs;
	std::vector<float> outputs;
};

// TODO: CUDA acceleration: https://www.youtube.com/watch?v=oQT7IC0x254&list=PLU0zjpa44nPXddA_hWV1U8oO7AevFgXnT
// TODO: Cross entropy
// TODO: Backpropagation momentum
// TODO: Softmax for output layer
// REF: https://www.youtube.com/watch?v=hfMk-kjRv4c

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func);
	NeuralNetwork(const std::string& path);

	void Train_Backpropagation();
	void Train_FiniteDifference();
	void Learn(float rate);
	std::vector<float> Forward(const std::vector<float>& params);

	void SetTrainingData(const array_2d_t& inputs, const array_2d_t& outputs);
	void SetStochastic(size_t batchesCount);
	float CalculateCost(const std::vector<NNBatch>& batches);
	void RandomizeLayers(float min, float max);

	std::vector<NNBatch> GetBatches() const;
	const NNData& GetData() const;
	void SaveToFile(const std::string& path) const;

private:
	NNData m_Data;
	NNData m_Gradient;
	NNActivationFunction m_Func;
	array_2d_t m_Inputs;
	array_2d_t m_Outputs;
	size_t m_BatchesCount;
	std::vector<Matrix> m_Layers;
	std::vector<Matrix> m_LayersRaw;
};