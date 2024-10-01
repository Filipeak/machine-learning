#include "NNUtils.h"
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
	return x > 0.0f ? x : 0.0f;
}

float NNActivationFuncs::ReLU_Derivative(float x)
{
	return x > 0.0f ? 1.0f : 0.0f;
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


Matrix::Matrix(size_t r, size_t c) : m_Rows(r), m_Columns(c)
{
	NN_ASSERT(r > 0);
	NN_ASSERT(c > 0);

	size_t totalSize = r * c;

	m_Data.reserve(totalSize);

	for (size_t i = 0; i < totalSize; i++)
	{
		m_Data.push_back(0);
	}
}

size_t Matrix::GetRowsSize() const
{
	return m_Rows;
}

size_t Matrix::GetColumnsSize() const
{
	return m_Columns;
}

float& Matrix::GetValueAtRef(size_t row, size_t col)
{
	return m_Data[GetArrayIndex(row, col)];
}

float Matrix::GetValueAt(size_t row, size_t col) const
{
	return m_Data[GetArrayIndex(row, col)];
}

size_t Matrix::GetArrayIndex(size_t r, size_t c) const
{
	NN_ASSERT(r < m_Rows && r >= 0);
	NN_ASSERT(c < m_Columns && c >= 0);

	return r * m_Columns + c;
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
	NN_ASSERT(a.GetRowsSize() == b.GetRowsSize());
	NN_ASSERT(a.GetColumnsSize() == b.GetColumnsSize());

	Matrix result(a.GetRowsSize(), a.GetColumnsSize());

	for (size_t i = 0; i < result.GetRowsSize(); i++)
	{
		for (size_t j = 0; j < result.GetColumnsSize(); j++)
		{
			result.GetValueAtRef(i, j) = a.GetValueAt(i, j) + b.GetValueAt(i, j);
		}
	}

	return result;
}

Matrix operator*(const Matrix& a, const Matrix& b)
{
	NN_ASSERT(a.GetColumnsSize() == b.GetRowsSize());

	Matrix result(a.GetRowsSize(), b.GetColumnsSize());

	for (size_t i = 0; i < a.GetRowsSize(); i++)
	{
		for (size_t j = 0; j < b.GetColumnsSize(); j++)
		{
			float sum = 0.0f;

			for (size_t k = 0; k < a.GetColumnsSize(); k++)
			{
				sum += a.GetValueAt(i, k) * b.GetValueAt(k, j);
			}

			result.GetValueAtRef(i, j) = sum;
		}
	}

	return result;
}

void NNData::Alloc(const std::vector<size_t>& architecture)
{
	arch = architecture;

	weights.reserve(architecture.size() - 1);
	biases.reserve(architecture.size() - 1);

	for (size_t i = 1; i < architecture.size(); i++)
	{
		weights.emplace_back(architecture[i], architecture[i - 1]);
		biases.emplace_back(architecture[i], 1);
	}
}