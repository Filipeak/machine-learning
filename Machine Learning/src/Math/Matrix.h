#pragma once

#include <vector>
#include <iostream>

typedef std::vector<std::vector<float>> array_2d_t;

class Matrix
{
public:
	Matrix();
	Matrix(size_t r, size_t c);

	size_t GetRowsSize() const;
	size_t GetColumnsSize() const;
	float& ValueAt(size_t row, size_t col);
	float GetValueAt(size_t row, size_t col) const;

private:
	array_2d_t m_Data;
};

std::ostream& operator<<(std::ostream& out, Matrix& mat);
Matrix operator+(const Matrix& a, const Matrix& b);
Matrix operator*(const Matrix& a, const Matrix& b);