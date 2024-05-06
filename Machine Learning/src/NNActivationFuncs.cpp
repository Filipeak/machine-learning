#include "NNActivationFuncs.h"
#include <cmath>

float NNActivationFuncs::CallFunc(NNActivationFunction func, float x)
{
	switch (func)
	{
	case NNActivationFunction::Sigmoid:
		return Sigmoid(x);
	case NNActivationFunction::ReLU:
		return ReLU(x);
	case NNActivationFunction::Tanh:
		return Tanh(x);
	default:
		return 0;
	}
}

float NNActivationFuncs::CallFuncDerivative(NNActivationFunction func, float x)
{
	switch (func)
	{
	case NNActivationFunction::Sigmoid:
		return Sigmoid_Derivative(x);
	case NNActivationFunction::ReLU:
		return ReLU_Derivative(x);
	case NNActivationFunction::Tanh:
		return Tanh_Derivative(x);
	default:
		return 0;
	}
}

float NNActivationFuncs::Sigmoid(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

float NNActivationFuncs::Sigmoid_Derivative(float x)
{
	return Sigmoid(x) * (1.0f - Sigmoid(x));
}

float NNActivationFuncs::ReLU(float x)
{
	return x > 0 ? x : 0;
}

float NNActivationFuncs::ReLU_Derivative(float x)
{
	return x > 0 ? 1 : 0;
}

float NNActivationFuncs::Tanh(float x)
{
	return tanhf(x);
}

float NNActivationFuncs::Tanh_Derivative(float x)
{
	float t = Tanh(x);

	return 1.0f - t * t;
}