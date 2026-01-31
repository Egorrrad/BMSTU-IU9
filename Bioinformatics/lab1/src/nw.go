package main

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"
)

const (
	UP uint8 = 1 << iota
	LEFT
	DIAG

	gapSymbol uint8 = '_'
)

func scoreFun(a, b uint8, matchScore, mismatchScore int) int {
	if a == b {
		return matchScore
	}
	return mismatchScore
}

/*
NeedlemanWunsch

Given two sequences, aligns them using the Needleman-Wunsch algorithm.

	This function takes two sequences and optionally a scoring function and a
	gap penalty value as arguments.
	The function returns a tuple containing the optimal alignment score and the
	aligned sequences, e.g. (10, 'ACCGT', 'AC-GT').

	Args:
	    seq1: The first sequence, e.g. 'ACCGT'
	    seq2: The second sequence, e.g. 'ACGT'
	    score_fun: The scoring function, e.g. score_fun('A', 'A') returns 5
	    gap_penalty: The gap penalty value, e.g. -10

	Returns:
	    score: The optimal alignment score, e.g. 10
	    aligned_seq1: The first aligned sequence, e.g. 'ACCGT'
	    aligned_seq2: The second aligned sequence, e.g. 'AC-GT'
*/
func NeedlemanWunsch(seq1, seq2 string, scoreFunc func(a, b uint8, matchScore, mismatchScore int) int, gapPenalty int) (int, string, string) {
	n := len(seq1)
	m := len(seq2)

	d := make([][]int, n+1)
	ptrD := make([][]uint8, n+1)
	for i := range d {
		d[i] = make([]int, m+1)
		ptrD[i] = make([]uint8, m+1)
	}

	// инициализируем первую строку и первый столбец
	d[0][0] = 0
	for i := 1; i <= n; i++ {
		d[i][0] = i * gapPenalty
	}

	for j := 1; j <= m; j++ {
		d[0][j] = j * gapPenalty
	}

	// заполняем матрицу по рекурентным соотношениям
	for i := 1; i <= n; i++ {
		el1 := seq1[i-1]
		for j := 1; j <= m; j++ {
			el2 := seq2[j-1]
			s := scoreFunc(el1, el2, 0, 0)
			s1, s2, s3 := d[i-1][j]+gapPenalty, d[i][j-1]+gapPenalty, d[i-1][j-1]+s
			m := math.Max(float64(s1), float64(s2))
			m = math.Max(m, float64(s3))
			sm := int(m)
			d[i][j] = sm

			// запоминаем направление движения
			switch sm {
			case s1:
				ptrD[i][j] = UP
			case s2:
				ptrD[i][j] = LEFT
			case s3:
				ptrD[i][j] = DIAG
			}

		}
	}

	if Debug {
		printArray(d)
	}

	// востановление оптимально выровненных последовательностей
	score := d[n][m]
	a := strings.Builder{}
	b := strings.Builder{}

	i, j := n, m

	for {
		if i == 0 && j == 0 {
			break
		}

		switch ptrD[i][j] {
		case UP:
			// идем вверх и добавляем в строку b символ гепа
			i--
			a.WriteByte(seq1[i])
			b.WriteByte(gapSymbol)
		case LEFT:
			// идем влево и добавляем в строку a символ гепа
			j--
			a.WriteByte(gapSymbol)
			b.WriteByte(seq2[j])
		case DIAG:
			// просто идем по диагонали
			i--
			j--
			a.WriteByte(seq1[i])
			b.WriteByte(seq2[j])
		default:
			// Если нет указателя куда идти (крайние случаи)
			if i > 0 && j == 0 {
				i--
				a.WriteByte(seq1[i])
				b.WriteByte(gapSymbol)
			} else if j > 0 && i == 0 {
				j--
				a.WriteByte(gapSymbol)
				b.WriteByte(seq2[j])
			} else {
				break
			}
		}
	}

	// разворачиваем строки
	return score, ReadFromRightReverse(a.String()), ReadFromRightReverse(b.String())
}

// функция для реверса строк
func ReadFromRightReverse(s string) string {
	r := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		r[i] = s[len(s)-1-i]
	}
	return string(r)
}

func printArray(matrix [][]int) {
	for _, row := range matrix {
		for _, el := range row {
			fmt.Printf("%6d", el)
		}
		fmt.Println()
	}
}

/*
Prints the results of the Needleman-Wunsch algorithm.

	This function takes two aligned sequences and the optimal alignment score
	as arguments. It prints the sequences and the score to the standard output
	or to a file.

	Args:
	    seq1: The first aligned sequence, e.g. 'ACCGT'
	    seq2: The second aligned sequence, e.g. 'AC-GT'
	    score: The optimal alignment score, e.g. 10
	    file: The file to print to. If None, prints to the standard output.

	Returns:
	    None
*/
func printResults(seq1, seq2 string, score int, file *os.File) {
	if file == nil {
		file = os.Stdout
	}

	// используем буферизированный вывод
	w := bufio.NewWriter(file)
	defer w.Flush()

	PrintMaxLen = len(seq1)
	printSubseq := func(i int, n, s string) {
		str := fmt.Sprintf("%s: %s\n", n, s[i:i+PrintMaxLen])
		w.WriteString(str)
	}

	w.WriteString("Pairwise alignment:\n")
	for i := 0; i < len(seq1); i += PrintMaxLen {
		printSubseq(i, "seq1", seq1)
		printSubseq(i, "seq2", seq2)
		w.WriteString("\n")
	}
	str := fmt.Sprintf("Score: %d\n", score)
	w.WriteString(str)

}
