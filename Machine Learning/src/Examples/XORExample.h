#pragma once

#include "ExampleBase.h"
#include "../Algorithms/NeuralNetwork.h"

class XORExample : public ExampleBase
{
public:
	XORExample() : m_NN(0) {}

	void Prepare() override;
	void RunIteration() override;

	float GetCost() override;
	const NNData& GetData() const;

private:
	NeuralNetwork* m_NN;
};