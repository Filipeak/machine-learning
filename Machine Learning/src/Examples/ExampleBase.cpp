#include "ExampleBase.h"

void ExampleBase::InitBackprop(float learningRate, size_t iterations)
{
	m_Backpropagation = true;
	m_LearningRate = learningRate;
	m_MaxIterations = iterations;
}

void ExampleBase::InitGradDescent(float learningRate, float derivativeEps, size_t iterations)
{
	m_Backpropagation = false;
	m_LearningRate = learningRate;
	m_DerivativeEps = derivativeEps;
	m_MaxIterations = iterations;
}

bool ExampleBase::IsFinished() const
{
	return m_Finished;
}