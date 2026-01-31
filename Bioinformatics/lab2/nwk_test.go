package main

import (
	"testing"
)

type UserData struct {
	seq1     string
	seq2     string
	match    int
	mismatch int
	gap      int
}

type ExpectedData struct {
	score   int
	algSeq1 string
	algSeq2 string
}

type TestCase struct {
	TestName string
	Data     *UserData
	ExpData  *ExpectedData
}

func TestNeedlemanWunsch(t *testing.T) {
	cases := []TestCase{

		// базовый тест из примера
		{
			TestName: "base test",
			Data:     &UserData{seq1: "ACGT", seq2: "ACGT", match: 5, mismatch: -4, gap: -10},
			ExpData:  &ExpectedData{score: 20, algSeq1: "ACGT", algSeq2: "ACGT"}},

		// тест с одной буквой
		{
			TestName: "test with one letter",
			Data:     &UserData{seq1: "A", seq2: "A", match: 5, mismatch: -4, gap: -10},
			ExpData:  &ExpectedData{score: 5, algSeq1: "A", algSeq2: "A"}},
	}

	for _, item := range cases {
		t.Run(item.TestName, func(t *testing.T) {
			data := item.Data
			expData := item.ExpData

			n, m := len(data.seq1), len(data.seq2)

			// тестируем обе реализации хранилища
			matrices := []MatrixStorage{
				NewFullMatrix(n, m),
				NewBandedMatrix(n, m),
			}

			for i, matrix := range matrices {
				matrixType := "FullMatrix"
				if i == 1 {
					matrixType = "BandedMatrix"
				}
				score := NeedlemanWunsch(data.seq1, data.seq2, matrix, data.match, data.mismatch, data.gap)

				if score != expData.score {
					t.Errorf("%s: Expected score %d, got %d", matrixType, expData.score, score)
				}
			}
		})
	}

	// тест для двух букв A вначале
	t.Run("test with two 'A' in the head", func(t *testing.T) {
		data := &UserData{seq1: "AACGT", seq2: "ACGT", match: 5, mismatch: -4, gap: -10}

		n, m := len(data.seq1), len(data.seq2)

		// тестируем обе реализации хранилища
		matrices := []MatrixStorage{
			NewFullMatrix(n, m),
			NewBandedMatrix(n, m),
		}

		for i, matrix := range matrices {
			matrixType := "FullMatrix"
			if i == 1 {
				matrixType = "BandedMatrix"
			}

			score := NeedlemanWunsch(data.seq1, data.seq2, matrix, data.match, data.mismatch, data.gap)
			if score != 10 {
				t.Errorf("%s: Expected score %d, got %d", matrixType, 10, score)
			}
		}
	})

}

func TestFullAndBandedMatrixConsistency(t *testing.T) {
	testCases := []struct {
		name string
		seq1 string
		seq2 string
	}{
		{"identical", "ACGT", "ACGT"},
		{"different", "ACGT", "TGCA"},
		{"medium", "ACGTAC", "ACGTAC"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			match, mismatch, gap := 5, -4, -10
			n := len(tc.seq1)

			fullMatrix := NewFullMatrix(n, n)
			bandedMatrix := NewBandedMatrix(n, n)

			scoreFull := NeedlemanWunsch(tc.seq1, tc.seq2, fullMatrix, match, mismatch, gap)
			scoreBanded := NeedlemanWunsch(tc.seq1, tc.seq2, bandedMatrix, match, mismatch, gap)

			if scoreFull != scoreBanded {
				t.Errorf("Scores differ: FullMatrix=%d, BandedMatrix=%d", scoreFull, scoreBanded)
			}
		})
	}
}

func TestKAndKPlus1Comparison(t *testing.T) {
	testCases := []struct {
		name string
		seq1 string
		seq2 string
	}{
		{"identical", "ACGT", "ACGT"},
		{"different", "ACGT", "TGCA"},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			match, mismatch, gap := 5, -4, -10
			n := len(tc.seq1)

			for k := 0; k < n && k < 3; k++ {
				matrixK := NewBandedMatrix(n, k)
				matrixKPlus1 := NewBandedMatrix(n, k+1)

				scoreK := NeedlemanWunsch(tc.seq1, tc.seq2, matrixK, match, mismatch, gap)
				scoreKPlus1 := NeedlemanWunsch(tc.seq1, tc.seq2, matrixKPlus1, match, mismatch, gap)

				if scoreKPlus1 < scoreK {
					t.Errorf("Score decreased from %d to %d when k increased from %d to %d",
						scoreK, scoreKPlus1, k, k+1)
				}
			}
		})
	}
}

func TestBandWidthLimitations(t *testing.T) {
	testCases := []struct {
		name string
		seq1 string
		seq2 string
		k    int
	}{
		{
			name: "narrow band fails",
			seq1: "AAAA",
			seq2: "TTTT",
			k:    0,
		},
		{
			name: "wide band works",
			seq1: "AAAA",
			seq2: "TTTT",
			k:    4,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			match, mismatch, gap := 5, -4, -10
			n := len(tc.seq1)

			optimalMatrix := NewFullMatrix(n, n)
			optimalScore := NeedlemanWunsch(tc.seq1, tc.seq2, optimalMatrix, match, mismatch, gap)

			limitedMatrix := NewBandedMatrix(n, tc.k)
			limitedScore := NeedlemanWunsch(tc.seq1, tc.seq2, limitedMatrix, match, mismatch, gap)

			if limitedScore != optimalScore && tc.k == n {
				t.Errorf("Expected optimal score %d with k=%d, but got %d",
					optimalScore, tc.k, limitedScore)
			}
		})
	}
}
