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
		{TestName: "base test",
			Data:    &UserData{seq1: "ACGT", seq2: "ACGT", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 20, algSeq1: "ACGT", algSeq2: "ACGT"}},

		// тест с одной буквой
		{TestName: "test with one letter", Data: &UserData{seq1: "A", seq2: "A", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 5, algSeq1: "A", algSeq2: "A"}},

		// тест с длиной a меньше длины b (и наоборот) в конце последовательности
		{TestName: "test with a < b (tail)", Data: &UserData{seq1: "ACGT", seq2: "ACGTQ", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 10, algSeq1: "ACGT_", algSeq2: "ACGTQ"}},
		{TestName: "test with a > b (tail)", Data: &UserData{seq1: "ACGTQR", seq2: "ACGTQ", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 15, algSeq1: "ACGTQR", algSeq2: "ACGTQ_"}},

		// тест с длиной a меньше длины b (и наоборот) в начале последовательности
		{TestName: "test with a < b (head)", Data: &UserData{seq1: "ACGT", seq2: "QACGT", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 10, algSeq1: "_ACGT", algSeq2: "QACGT"}},
		{TestName: "test with a > b (head)", Data: &UserData{seq1: "QACGT", seq2: "ACGT", match: 5, mismatch: -4, gap: -10},
			ExpData: &ExpectedData{score: 10, algSeq1: "QACGT", algSeq2: "_ACGT"}},
	}

	for _, item := range cases {
		t.Log("TestCase: ", item.TestName)
		data := item.Data
		expData := item.ExpData

		scoreFun := func(x, y uint8, matchScore, mismatchScore int) int {
			if x == y {
				return data.match
			}
			return data.mismatch
		}
		score, alignedSeq1, alignedSeq2 := NeedlemanWunsch(data.seq1, data.seq2, scoreFun, data.gap)

		if score != expData.score {
			t.Errorf("Expected score %d, got %d", expData.score, score)
		}
		if alignedSeq1 != expData.algSeq1 {
			t.Errorf("Expected alignedSeq1 '%s', got '%s'", expData.algSeq1, alignedSeq1)
		}
		if alignedSeq2 != expData.algSeq2 {
			t.Errorf("Expected alignedSeq2 '%s', got '%s'", expData.algSeq2, alignedSeq2)
		}
	}

	// тест для двух букв A вначале
	t.Log("TestCase: ", "test with two 'A' in the head")
	data := &UserData{seq1: "AACGT", seq2: "ACGT", match: 5, mismatch: -4, gap: -10}
	scoreFun := func(x, y uint8, matchScore, mismatchScore int) int {
		if x == y {
			return data.match
		}
		return data.mismatch
	}
	score, alignedSeq1, alignedSeq2 := NeedlemanWunsch(data.seq1, data.seq2, scoreFun, data.gap)
	if score != 10 {
		t.Errorf("Expected score %d, got %d", 10, score)
	}
	if alignedSeq1 != "AACGT" {
		t.Errorf("Expected alignedSeq1 '%s', got '%s'", "AACGT", alignedSeq1)
	}
	if alignedSeq2 != "_ACGT" && alignedSeq2 != "A_CGT" {
		t.Errorf("Expected alignedSeq2 '%s' or '%s', got '%s'", "_ACGT", "A_CGT", alignedSeq2)
	}
}
