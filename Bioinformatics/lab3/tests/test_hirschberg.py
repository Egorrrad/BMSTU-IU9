import hirschberg.align as align

def test_hirschberg_1():
    aln1, aln2, score = align.hirschberg("ACGT", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACGT"
    assert score == 20

def test_hirschberg_2():
    aln1, aln2, score = align.hirschberg("ACG", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACG-"
    assert aln2 == "ACGT"
    assert score == 10

def test_hirschberg_3():
    aln1, aln2, score = align.hirschberg("ACGT", "ACG")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACGT"
    assert aln2 == "ACG-"
    assert score == 10

def test_hirschberg_4():
    aln1, aln2, score = align.hirschberg("ACAGT", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACAGT"
    assert aln2 == "AC-GT"
    assert score == 15

def test_hirschberg_5():
    aln1, aln2, score = align.hirschberg("ACGT", "ACAGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "AC-GT"
    assert aln2 == "ACAGT"
    assert score == 15

def test_hirschberg_6():
    aln1, aln2, score = align.hirschberg("CAGT", "ACAGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "-CAGT"
    assert aln2 == "ACAGT"
    assert score == 15

def test_hirschberg_7():
    aln1, aln2, score = align.hirschberg("ACAGT", "CAGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACAGT"
    assert aln2 == "-CAGT"
    assert score == 15

def test_hirschberg_8():
    aln1, aln2, score = align.hirschberg("ACGT", "A")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACGT"
    assert aln2 == "A---"
    assert score == -10

def test_hirschberg_9():
    aln1, aln2, score = align.hirschberg("ACGT", "")
    assert len(aln1) == len(aln2)
    assert aln1 == "ACGT"
    assert aln2 == "----"
    assert score == -20

def test_hirschberg_10():
    aln1, aln2, score = align.hirschberg("A", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "A---"
    assert aln2 == "ACGT"
    assert score == -10

def test_hirschberg_11():
    aln1, aln2, score = align.hirschberg("", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "----"
    assert aln2 == "ACGT"
    assert score == -20

def test_hirschberg_12():
    aln1, aln2, score = align.hirschberg("", "")
    assert aln1 == ""
    assert aln2 == ""
    assert score == 0

def test_hirschberg_13():
    aln1, aln2, score = align.hirschberg("TACGT", "ATGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "TACGT"
    assert aln2 == "-ATGT"
    assert score == 6

def test_hirschberg_14():
    aln1, aln2, score = align.hirschberg("TACGT", "ACTGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "TAC-GT"
    assert aln2 == "-ACTGT"
    assert score == 10

def test_hirschberg_15():
    aln1, aln2, score = align.hirschberg("ACGT", "TAGTA")
    assert len(aln1) == len(aln2)
    assert aln1 == "-ACGT-"
    assert aln2 == "TA-GTA"
    assert score == 0

def test_hirschberg_16():
    aln1, aln2, score = align.hirschberg("TAGTA", "ACGT")
    assert len(aln1) == len(aln2)
    assert aln1 == "TA-GTA"
    assert aln2 == "-ACGT-"
    assert score == 0

def test_hirschberg_17():
    aln1, aln2, score = align.hirschberg("ACGT", "TAGT", gap_score=0)
    assert len(aln1) == len(aln2)
    assert aln1 == "-ACGT"
    assert aln2 == "TA-GT"
    assert score == 15

def test_hirschberg_18():
    aln1, aln2, score = align.hirschberg("TAGT", "ACGT", gap_score=10)
    assert len(aln1) == len(aln2)
    assert len(aln1) == 8
    assert score == 80


def test_hirschberg_vs_needleman_wunsch_identical():
    """
    Тест: сравнение алгоритмов на идентичных последовательностях.
    """
    seq1 = "ACGTACGT"
    seq2 = "ACGTACGT"

    aln1_h, aln2_h, score_h = align.hirschberg(seq1, seq2)
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch(seq1, seq2)

    assert aln1_h == aln1_nw, f"Aln1 mismatch: Hirschberg={aln1_h}, NW={aln1_nw}"
    assert aln2_h == aln2_nw, f"Aln2 mismatch: Hirschberg={aln2_h}, NW={aln2_nw}"
    assert score_h == score_nw, f"Score mismatch: Hirschberg={score_h}, NW={score_nw}"
    assert aln1_h == seq1, "Aligned sequence should match original"
    assert aln2_h == seq2, "Aligned sequence should match original"


def test_hirschberg_vs_needleman_wunsch_with_gaps():
    """
    Тест: сравнение алгоритмов на последовательностях с делениями.

    Проверяет корректность обработки внутренних делений. Алгоритм
    Хиршберга должен правильно находить оптимальные позиции для
    вставки гэпов, как и классический NW.
    """
    seq1 = "ACGTACGT"
    seq2 = "ACGTCGT"

    aln1_h, aln2_h, score_h = align.hirschberg(seq1, seq2)
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch(seq1, seq2)

    assert aln1_h == aln1_nw, f"Aln1 mismatch for sequences with gap"
    assert aln2_h == aln2_nw, f"Aln2 mismatch for sequences with gap"
    assert score_h == score_nw, f"Score mismatch for sequences with gap"

    assert aln1_h.replace('-', '') == seq1
    assert aln2_h.replace('-', '') == seq2


def test_hirschberg_vs_needleman_wunsch_completely_different():
    """
    Тест: сравнение на полностью различных последовательностях.

    Демонстрирует корректность работы алгоритмов в крайнем случае,
    когда последовательности не имеют общих символов. Оба алгоритма
    должны найти выравнивание с минимальным скором.
    """
    seq1 = "AAAA"
    seq2 = "TTTT"

    aln1_h, aln2_h, score_h = align.hirschberg(seq1, seq2)
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch(seq1, seq2)

    assert aln1_h == aln1_nw
    assert aln2_h == aln2_nw
    assert score_h == score_nw

    # низкий скор из-за полного несовпадения
    expected_score = 4 * align.score_fun('A', 'T')  # 4 несовпадения
    assert score_h == expected_score


def test_hirschberg_vs_needleman_wunsch_custom_scoring():
    """
    Тест: сравнение с пользовательской функцией оценки.

    Проверяет, что алгоритм Хиршберга корректно работает с разными
    параметрами скоринга, включая нестандартные значения match/mismatch
    и gap penalty.
    """
    seq1 = "ACGT"
    seq2 = "ACCT"

    def custom_score(a, b):
        if a == b:
            return 10
        else:
            return -7

    aln1_h, aln2_h, score_h = align.hirschberg(seq1, seq2, score_fun=custom_score, gap_score=-3)
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch(seq1, seq2, score_fun=custom_score, gap_score=-3)

    assert score_h == score_nw, f"Score mismatch with custom scoring: {score_h} vs {score_nw}"

    assert len(aln1_h) == len(aln2_h), "Alignments must have same length"
    assert aln1_h.replace('-', '') == seq1, "Original sequence 1 must be preserved"
    assert aln2_h.replace('-', '') == seq2, "Original sequence 2 must be preserved"


def test_hirschberg_memory_efficiency_demonstration():
    """
    Тест: демонстрация эффективности использования памяти.

    В этом тесте мы проверяем только совпадение скоров,
    так как для длинных последовательностей может существовать множество
    оптимальных выравниваний с одинаковым скором.
    """

    long_seq1 = "ACGT" * 25
    long_seq2 = "TGCA" * 25

    aln1_h, aln2_h, score_h = align.hirschberg(long_seq1, long_seq2)
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch(long_seq1, long_seq2)

    assert score_h == score_nw, f"Score mismatch for long sequences: {score_h} vs {score_nw}"

    assert len(aln1_h) == len(aln2_h), "Alignments must have same length"
    assert aln1_h.replace('-', '') == long_seq1, "Original sequence 1 must be preserved"
    assert aln2_h.replace('-', '') == long_seq2, "Original sequence 2 must be preserved"


def test_hirschberg_edge_case_single_char():
    """
    Тест: граничный случай с одним символом.

    Проверяет корректность работы алгоритма, когда одна или обе
    последовательности состоят из одного символа. Это важный
    граничный случай для рекурсивного алгоритма.
    """
    aln1_h, aln2_h, score_h = align.hirschberg("A", "T")
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch("A", "T")

    assert aln1_h == aln1_nw
    assert aln2_h == aln2_nw
    assert score_h == score_nw

    aln1_h, aln2_h, score_h = align.hirschberg("A", "TTT")
    aln1_nw, aln2_nw, score_nw = align.needleman_wunsch("A", "TTT")

    assert aln1_h == aln1_nw
    assert aln2_h == aln2_nw
    assert score_h == score_nw


def run_all_tests():
    test_functions = [
        test_hirschberg_1, test_hirschberg_2, test_hirschberg_3,
        test_hirschberg_4, test_hirschberg_5, test_hirschberg_6,
        test_hirschberg_7, test_hirschberg_8, test_hirschberg_9,
        test_hirschberg_10, test_hirschberg_11, test_hirschberg_12,
        test_hirschberg_13, test_hirschberg_14, test_hirschberg_15,
        test_hirschberg_16, test_hirschberg_17, test_hirschberg_18,
        test_hirschberg_vs_needleman_wunsch_identical,
        test_hirschberg_vs_needleman_wunsch_with_gaps,
        test_hirschberg_vs_needleman_wunsch_completely_different,
        test_hirschberg_vs_needleman_wunsch_custom_scoring,
        test_hirschberg_memory_efficiency_demonstration,
        test_hirschberg_edge_case_single_char
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__} passed")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__} failed: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__} error: {e}")
            failed += 1

    print(f"\nTotal: {passed} passed, {failed} failed, {passed + failed} total")

    if failed == 0:
        print("All tests passed!")
    else:
        print("Some tests failed")


if __name__ == "__main__":
    run_all_tests()