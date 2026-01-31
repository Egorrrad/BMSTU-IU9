package main

import (
	"fmt"
)

const (
	UP uint8 = 1 << iota
	LEFT
	DIAG
)

func scoreFun(a, b uint8, matchScore, mismatchScore int) int {
	if a == b {
		return matchScore
	}
	return mismatchScore
}

/*
NeedlemanWunsch выравнивает последовательности с использованием ограниченной ширины диагонали

	Args:
	    seq1, seq2: выравниваемые последовательности
	    matrix: реализация MatrixStorage для хранения матрицы
	    match, mismatch, gap: параметры скоринга

	Returns:
	    score: оценка выравнивания
*/
func NeedlemanWunsch(seq1, seq2 string, matrix MatrixStorage, match, mismatch, gap int) int {
	var score int
	n := len(seq1)
	m := len(seq2)

	// Инициализация матрицы указателей
	ptrMatrix := make([][]uint8, n+1)
	for i := range ptrMatrix {
		ptrMatrix[i] = make([]uint8, m+1)
	}

	// Инициализация первой строки и столбца
	matrix.Set(0, 0, 0)
	for i := 1; i <= n; i++ {
		if matrix.IsValid(i, 0) {
			matrix.Set(i, 0, i*gap)
			ptrMatrix[i][0] = UP
		}
	}

	for j := 1; j <= m; j++ {
		if matrix.IsValid(0, j) {
			matrix.Set(0, j, j*gap)
			ptrMatrix[0][j] = LEFT
		}
	}

	// Заполнение матрицы
	for i := 1; i <= n; i++ {
		el1 := seq1[i-1]
		for j := 1; j <= m; j++ {
			if !matrix.IsValid(i, j) {
				continue // Пропускаем ячейки вне полосы
			}

			// Вычисляем скоринговые значения только для допустимых ячеек
			diagScore := minValue
			if matrix.IsValid(i-1, j-1) {
				s := scoreFun(el1, seq2[j-1], match, mismatch)
				diagScore = matrix.Get(i-1, j-1) + s
			}

			upScore := minValue
			if matrix.IsValid(i-1, j) {
				upScore = matrix.Get(i-1, j) + gap
			}

			leftScore := minValue
			if matrix.IsValid(i, j-1) {
				leftScore = matrix.Get(i, j-1) + gap
			}

			// Находим максимальное значение
			maxScore := maxInt(diagScore, upScore, leftScore)
			matrix.Set(i, j, maxScore)

			// Запоминаем направление
			switch maxScore {
			case diagScore:
				ptrMatrix[i][j] = DIAG
			case upScore:
				ptrMatrix[i][j] = UP
			case leftScore:
				ptrMatrix[i][j] = LEFT
			}
		}
	}

	if Debug {
		fmt.Println("Scoring matrix:")
		fmt.Println(matrix)
	}

	score = matrix.Get(n, m)
	return score
}
