#pragma once

#include "../Math/Matrix.h"

class ExampleBase
{
public:
	ExampleBase() : m_Finished(false), m_Backpropagation(true), m_LearningRate(0), m_MaxIterations(0), m_CurrentIteration(0) {}

	void InitBackprop(float learningRate, size_t iterations);
	void InitFiniteDiff(float learningRate, size_t iterations);

	bool IsFinished() const;

	virtual void Prepare() = 0;
	virtual void RunIteration() = 0;
	virtual float GetCost() = 0;

protected:
	bool m_Finished;
	bool m_Backpropagation;
	float m_LearningRate;
	size_t m_MaxIterations;
	size_t m_CurrentIteration;
};