package main

import (
	"fmt"
	"strings"
)

type FullMatrix struct {
	n      int
	k      int
	matrix [][]int
}

// NewFullMatrix создает FullMatrix заданного размера
func NewFullMatrix(n, k int) *FullMatrix {
	matrix := make([][]int, n+1)
	for i := range matrix {
		matrix[i] = make([]int, n+1)
		for j := range matrix[i] {
			matrix[i][j] = minValue // минимальное int
		}
	}
	return &FullMatrix{
		n:      n,
		k:      k,
		matrix: matrix,
	}
}

// получение значения ячейки
func (m *FullMatrix) Get(i, j int) int {
	if !m.IsValid(i, j) {
		return minValue // для ячеек вне полосы
	}
	return m.matrix[i][j]
}

// установление значения ячейки
func (m *FullMatrix) Set(i, j int, v int) {
	if m.IsValid(i, j) {
		m.matrix[i][j] = v
	}
	// если ячейка вне полосы, то игнорируем установку
}

// проверка, находится ли ячейка в допустимой области
func (m *FullMatrix) IsValid(i, j int) bool {
	return i >= 0 && i <= m.n && j >= 0 && j <= m.k && abs(i-j) <= m.k
}

// функция для вывода матрицы
func (m *FullMatrix) String() string {
	res := strings.Builder{}
	for _, row := range m.matrix {
		for _, el := range row {
			res.WriteString(fmt.Sprintf("%6d", el))
		}
		res.WriteString("\n")
	}
	return res.String()
}
