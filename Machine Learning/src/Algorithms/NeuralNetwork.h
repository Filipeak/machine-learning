#pragma once

#include <vector>
#include <string>
#include <functional>

#include "../Math/Matrix.h"

#define DERIVATIVE_EPS 0.001f;

enum class NNActivationFunction
{
	Sigmoid = 0,
	ReLU = 1,
	Tanh = 2,
};

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

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func);

	void Train_Backpropagation();
	void Train_FiniteDifference();

	void SetTrainingData(const array_2d_t& inputs, const array_2d_t& outputs);
	void SetStochastic(size_t batchesCount);
	void Learn(float rate);
	std::vector<float> Forward(const std::vector<float>& params);
	float CalculateCost(const std::vector<NNBatch>& batches);
	void RandomizeLayers(float min, float max);

	std::vector<NNBatch> GetBatches() const;
	const NNData& GetData() const;
	void Print();

private:
	NNData m_Data;
	NNData m_Gradient;
	NNActivationFunction m_Func;
	array_2d_t m_Inputs;
	array_2d_t m_Outputs;
	size_t m_BatchesCount;
	std::vector<Matrix> m_Layers;
	std::vector<Matrix> m_LayersRaw;
	std::function<float(float)> m_ActivationFunction;
	std::function<float(float)> m_ActivationFunctionDerivative;

	void SetActivationFunctions(NNActivationFunction func);
	float RandomFloat01();

	static float Sigmoid(float x);
	static float Sigmoid_Derivative(float x);
	static float ReLU(float x);
	static float ReLU_Derivative(float x);
	static float Tanh(float x);
	static float Tanh_Derivative(float x);
};