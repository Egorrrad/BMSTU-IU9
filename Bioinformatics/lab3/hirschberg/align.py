from typing import Callable, Tuple

DEBUG = False


def score_fun(a: str, b: str, match_score: int = 5, mismatch_score: int = -4) -> int:
    return match_score if a == b else mismatch_score


def linmem_last_line(seq1: str, seq2: str, score_fun: Callable, gap_score: int) -> list:
    m, n = len(seq1), len(seq2)

    if m > n:
        return linmem_last_line(seq2, seq1, score_fun, gap_score)

    prev = [j * gap_score for j in range(n + 1)]
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i * gap_score
        for j in range(1, n + 1):
            match_score = prev[j - 1] + score_fun(seq1[i - 1], seq2[j - 1])
            delete_score = prev[j] + gap_score
            insert_score = curr[j - 1] + gap_score
            curr[j] = max(match_score, delete_score, insert_score)
        prev, curr = curr, prev

    return prev


def get_mid_j(left_last_line: list, right_last_line: list) -> int:
    min_len = min(len(left_last_line), len(right_last_line))
    left_last_line = left_last_line[:min_len]
    right_last_line = right_last_line[:min_len]

    max_score = float('-inf')
    mid_j = 0

    for j in range(len(left_last_line)):
        total_score = left_last_line[j] + right_last_line[j]
        if total_score > max_score:
            max_score = total_score
            mid_j = j

    return mid_j


def hirschberg(seq1: str, seq2: str, score_fun: Callable = score_fun, gap_score: int = -5) -> Tuple[str, str, int]:
    '''
    Inputs:
    seq1 - first sequence
    seq2 - second sequence
    score_fun - function that returns score for two symbols
    gap_score - score for gap in final alignment

    Outputs:
    aln1 - first sequence in alignment
    aln2 - second sequence in alignment
    score - score of alignment
    '''

    if len(seq1) == 0:
        return '-' * len(seq2), seq2, gap_score * len(seq2)
    if len(seq2) == 0:
        return seq1, '-' * len(seq1), gap_score * len(seq1)
    if len(seq1) == 1 or len(seq2) == 1:
        return needleman_wunsch(seq1, seq2, score_fun, gap_score)

    if len(seq1) <= len(seq2):

        mid_i = len(seq1) // 2

        left_last_line = linmem_last_line(seq1[:mid_i], seq2, score_fun, gap_score)
        right_last_line = linmem_last_line(seq1[mid_i:][::-1], seq2[::-1], score_fun, gap_score)
        right_last_line = right_last_line[::-1]

        mid_j = get_mid_j(left_last_line, right_last_line)

        left_aln1, left_aln2, left_score = hirschberg(seq1[:mid_i], seq2[:mid_j], score_fun, gap_score)
        right_aln1, right_aln2, right_score = hirschberg(seq1[mid_i:], seq2[mid_j:], score_fun, gap_score)
    else:

        mid_j = len(seq2) // 2

        top_last_line = linmem_last_line(seq2[:mid_j], seq1, score_fun, gap_score)
        bottom_last_line = linmem_last_line(seq2[mid_j:][::-1], seq1[::-1], score_fun, gap_score)
        bottom_last_line = bottom_last_line[::-1]

        mid_i = get_mid_j(top_last_line, bottom_last_line)

        left_aln1, left_aln2, left_score = hirschberg(seq1[:mid_i], seq2[:mid_j], score_fun, gap_score)
        right_aln1, right_aln2, right_score = hirschberg(seq1[mid_i:], seq2[mid_j:], score_fun, gap_score)

    aln1 = left_aln1 + right_aln1
    aln2 = left_aln2 + right_aln2
    score = left_score + right_score

    return aln1, aln2, score


def needleman_wunsch(seq1: str, seq2: str, score_fun: Callable = score_fun, gap_score: int = -5):
    m, n = len(seq1) + 1, len(seq2) + 1

    matrix = [[0] * n for _ in range(m)]

    for i in range(m):
        matrix[i][0] = i * gap_score
    for j in range(n):
        matrix[0][j] = j * gap_score

    for i in range(1, m):
        for j in range(1, n):
            matrix[i][j] = max(matrix[i - 1][j - 1] + score_fun(seq1[i - 1], seq2[j - 1]), matrix[i - 1][j] + gap_score,
                               matrix[i][j - 1] + gap_score)
    if DEBUG:
        print_array(matrix)

    score = matrix[-1][-1]
    i, j = m - 1, n - 1
    aln1 = ""
    aln2 = ""
    while i > 0 or j > 0:
        a, b = '-', '-'
        # (A, B)
        if i > 0 and j > 0 and matrix[i][j] == matrix[i - 1][j - 1] + score_fun(seq1[i - 1], seq2[j - 1]):
            a = seq1[i - 1]
            b = seq2[j - 1]
            i -= 1
            j -= 1

        # (A, -)
        elif i > 0 and matrix[i][j] == matrix[i - 1][j] + gap_score:
            a = seq1[i - 1]
            i -= 1

        # (-, A)
        elif j > 0 and matrix[i][j] == matrix[i][j - 1] + gap_score:
            b = seq2[j - 1]
            j -= 1

        aln1 += a
        aln2 += b
    return aln1[::-1], aln2[::-1], score


def print_array(matrix: list):
    for row in matrix:
        for element in row:
            print(f"{element:6}", end="")
        print()


if __name__ == "__main__":
    # aln1, aln2, score = hirschberg("ATCT", "ACT", gap_score=-5)
    aln1, aln2, score = needleman_wunsch("ATCT", "ACT", gap_score=-5)

    assert len(aln1) == len(aln2)
    print(aln1)
    print(aln2)
    print(score)
