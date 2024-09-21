#pragma once

#include <cassert>
#include <vector>

#define NN_ASSERT assert

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

class Matrix
{
public:
	Matrix(size_t r, size_t c);

	size_t GetRowsSize() const;
	size_t GetColumnsSize() const;
	float& GetValueAtRef(size_t row, size_t col);
	float GetValueAt(size_t row, size_t col) const;

private:
	size_t m_Rows;
	size_t m_Columns;
	std::vector<float> m_Data;

	size_t GetArrayIndex(size_t r, size_t c) const;
};

Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);

struct NNData
{
	std::vector<size_t> arch;
	std::vector<Matrix> weights;
	std::vector<Matrix> biases;

	void Alloc(const std::vector<size_t>& architecture);
};

struct NNBatch
{
	std::vector<float> inputs;
	std::vector<float> outputs;
};