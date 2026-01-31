package main

import "math"

const minValue = -1 << 31

type MatrixStorage interface {
	Get(i, j int) int
	Set(i, j int, v int)
	IsValid(i, j int) bool
}

// abs возвращает абсолютное значение целого числа
func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

// maxInt возвращает максимальное число типа int
func maxInt(a, b, c int) int {
	return int(math.Max(float64(a), math.Max(float64(b), float64(c))))
}
