#include "Matrix.h"
#include "../Utils/Assert.h"

Matrix::Matrix()
{
}

Matrix::Matrix(size_t r, size_t c)
{
	ASSERT(r > 0);
	ASSERT(c > 0);

	for (size_t i = 0; i < r; i++)
	{
		m_Data.push_back(std::vector<float>());

		for (size_t j = 0; j < c; j++)
		{
			m_Data[i].push_back(0.0f);
		}
	}
}

size_t Matrix::GetRowsSize() const
{
	return m_Data.size();
}

size_t Matrix::GetColumnsSize() const
{
	ASSERT(m_Data.size() > 0);

	return m_Data[0].size();
}

float& Matrix::ValueAt(size_t row, size_t col)
{
	ASSERT(row < m_Data.size() && row >= 0);
	ASSERT(col < m_Data[0].size() && col >= 0);

	return m_Data[row][col];
}

float Matrix::GetValueAt(size_t row, size_t col) const
{
	ASSERT(row < m_Data.size() && row >= 0);
	ASSERT(col < m_Data[0].size() && col >= 0);

	return m_Data[row][col];
}

std::ostream& operator<<(std::ostream& out, Matrix& mat)
{
	for (size_t i = 0; i < mat.GetRowsSize(); i++)
	{
		for (size_t j = 0; j < mat.GetColumnsSize(); j++)
		{
			out << mat.ValueAt(i, j) << " ";
		}

		out << std::endl;
	}

	return out;
}

Matrix operator+(const Matrix& a, const Matrix& b)
{
	ASSERT(a.GetRowsSize() == b.GetRowsSize());
	ASSERT(a.GetColumnsSize() == b.GetColumnsSize());

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
	ASSERT(a.GetColumnsSize() == b.GetRowsSize());

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