package main

import (
	"fmt"
	"strings"
)

type BandedMatrix struct {
	n      int
	k      int
	matrix [][]int
}

// NewBandedMatrix создает BandedMatrix заданного размера
func NewBandedMatrix(n, k int) *BandedMatrix {
	matrix := make([][]int, n+1)
	for i := range matrix {
		matrix[i] = make([]int, 2*k+1)
		for j := range matrix[i] {
			matrix[i][j] = minValue // минимальное значение
		}
	}
	return &BandedMatrix{
		n:      n,
		k:      k,
		matrix: matrix,
	}
}

// получение значения ячейки
func (b *BandedMatrix) Get(i, j int) int {
	if !b.IsValid(i, j) {
		return minValue // для ячеек вне полосы
	}
	// Преобразуем координаты в индекс полосы
	bandIndex := j - i + b.k
	return b.matrix[i][bandIndex]
}

// установление значения ячейки
func (b *BandedMatrix) Set(i, j int, v int) {
	if b.IsValid(i, j) {
		bandIndex := j - i + b.k
		b.matrix[i][bandIndex] = v
	}
}

// проверка, находится ли ячейка в допустимой области
func (b *BandedMatrix) IsValid(i, j int) bool {
	return i >= 0 && i <= b.n && j >= 0 && j <= b.n &&
		abs(i-j) <= b.k
}

// функция для вывода матрицы
func (b *BandedMatrix) String() string {
	res := strings.Builder{}
	for _, row := range b.matrix {
		for _, el := range row {
			res.WriteString(fmt.Sprintf("%6d", el))
		}
		res.WriteString("\n")
	}
	return res.String()
}
