#pragma once

#include <vector>
#include <string>
#include <functional>

#include "../Math/Matrix.h"

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

	void Alloc(const std::vector<size_t> architecture);
	void Clear();
};

class NeuralNetwork
{
public:
	NeuralNetwork(const std::vector<size_t>& architecture, NNActivationFunction func);

	void Train_Backpropagation(NNData& gradient, const array_2d_t& inputs, const array_2d_t& outputs);
	void Train_FiniteDifference(NNData& gradient, const array_2d_t& inputs, const array_2d_t& outputs, float eps);

	void Learn(NNData& gradient, float rate);
	std::vector<float> Forward(const std::vector<float>& params);
	float CalculateCost(const array_2d_t& inputs, const array_2d_t& outputs);
	void RandomizeLayers(float min, float max);

	const NNData& GetData() const;
	void Print();

private:
	NNData m_Data;
	NNActivationFunction m_Func;
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