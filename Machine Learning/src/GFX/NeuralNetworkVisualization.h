#pragma once

#include "../Algorithms/NeuralNetwork.h"

class NeuralNetworkVisualization
{
public:
	void Init();
	void Update(const NNData& nnData);
	void Terminate();

private:
	GLuint m_ShaderProgram, m_VAO, m_VBO;
};