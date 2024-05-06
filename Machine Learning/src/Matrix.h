#pragma once

#include <vector>

// TODO: Change vector to pointer - remove unnecessary copying, etc.

class Matrix
{
public:
	Matrix(size_t r, size_t c);

	size_t GetRowsSize() const;
	size_t GetColumnsSize() const;
	float& ValueAt(size_t row, size_t col);
	float GetValueAt(size_t row, size_t col) const;

private:
	size_t m_Rows;
	size_t m_Columns;
	std::vector<float> m_Data;
};

Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);