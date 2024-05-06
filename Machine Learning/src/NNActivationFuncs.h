#pragma once

enum class NNActivationFunction
{
	Sigmoid = 0,
	ReLU = 1,
	Tanh = 2,
};

class NNActivationFuncs
{
public:
	static float CallFunc(NNActivationFunction func, float x);
	static float CallFuncDerivative(NNActivationFunction func, float x);

private:
	static float Sigmoid(float x);
	static float Sigmoid_Derivative(float x);
	static float ReLU(float x);
	static float ReLU_Derivative(float x);
	static float Tanh(float x);
	static float Tanh_Derivative(float x);
};