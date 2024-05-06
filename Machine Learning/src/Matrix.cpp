#include "Matrix.h"
#include "NNAssert.h"

Matrix::Matrix(size_t r, size_t c) : m_Rows(r), m_Columns(c)
{
	NN_ASSERT(r > 0);
	NN_ASSERT(c > 0);

	for (size_t i = 0; i < r * c; i++)
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

float& Matrix::ValueAt(size_t row, size_t col)
{
	NN_ASSERT(row < m_Data.size() && row >= 0);
	NN_ASSERT(col < m_Data[0].size() && col >= 0);

	return m_Data[row * m_Columns + col];
}

float Matrix::GetValueAt(size_t row, size_t col) const
{
	NN_ASSERT(row < m_Data.size() && row >= 0);
	NN_ASSERT(col < m_Data[0].size() && col >= 0);

	return m_Data[row * m_Columns + col];
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
			result.ValueAt(i, j) = a.GetValueAt(i, j) + b.GetValueAt(i, j);
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

			result.ValueAt(i, j) = sum;
		}
	}

	return result;
}