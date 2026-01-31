from typing import Callable, Tuple

DEBUG = False

def score_fun(a: str, 
              b: str,
              match_score: int = 5, 
              mismatch_score: int = -4) -> int:
    return match_score if a == b else mismatch_score

def needleman_wunsch_affine(seq1: str, 
                            seq2: str, 
                            score_fun: Callable = score_fun, 
                            gap_open: int = -10, 
                            gap_extend: int = -1) -> Tuple[str, str, int]:
    '''
    Inputs:
    seq1 - first sequence
    seq2 - second sequence
    score_fun - function that takes two characters and returns score
    gap_open - gap open penalty
    gap_extend - gap extend penalty
    Outputs:
    aln1 - first aligned sequence
    aln2 - second aligned sequence
    score - score of the alignment
    '''

    n = len(seq1)
    m = len(seq2)

    #infinity = 2 * gap_open + (n + m - 2) * gap_extend + 1
    infinity = float('-inf')

    # 1. Initialize matrices
    M = [[infinity for _ in range(m + 1)] for _ in range(n + 1)]
    I = [[infinity for _ in range(m + 1)] for _ in range(n + 1)]
    D = [[infinity for _ in range(m + 1)] for _ in range(n + 1)]
    # выравнивание двух пустых строк имеет штраф 0
    M[0][0] = 0

    for i in range(1, n + 1):
        D[i][0] = gap_open + (i - 1) * gap_extend

    for j in range(1, m + 1):
        I[0][j] = gap_open + (j - 1) * gap_extend

    # 2. Fill matrices
    # We assume that consecutive gaps on different sequences are not allowed
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = score_fun(seq1[i - 1], seq2[j - 1])


            I[i][j] = max(I[i][j - 1] + gap_extend,
                          M[i][j - 1] + gap_open,
                          D[i][j - 1] + gap_open)

            D[i][j] = max(D[i - 1][j] + gap_extend,
                          M[i - 1][j] + gap_open,
                          I[i - 1][j] + gap_open)

            M[i][j] = max(M[i - 1][j - 1] + s,
                          I[i - 1][j - 1] + s,
                          D[i - 1][j - 1] + s)

    score = max(M[n][m], I[n][m], D[n][m])

    # 3. Traceback
    aln1 = ''
    aln2 = ''
    i = n
    j = m

    # Определяем, с какой матрицы начать восстановление пути
    if score == M[n][m]:
        state = 'M'
    elif score == I[n][m]:
        state = 'I'
    else:
        state = 'D'

    while i > 0 or j > 0:
        if state == 'M':
            # Диагональный шаг: берем символы из обеих строк
            aln1 += seq1[i - 1]
            aln2 += seq2[j - 1]

            s = score_fun(seq1[i - 1], seq2[j - 1])
            current_val = M[i][j]

            if i > 0 and j > 0 and current_val == M[i - 1][j - 1] + s:
                state = 'M'
            elif i > 0 and j > 0 and current_val == I[i - 1][j - 1] + s:
                state = 'I'
            elif i > 0 and j > 0 and current_val == D[i - 1][j - 1] + s:
                state = 'D'
            else:
                state = 'M'

            i -= 1
            j -= 1

        elif state == 'I':
            # Горизонтальный шаг: гэп в seq1
            aln1 += '-'
            aln2 += seq2[j - 1]

            current_val = I[i][j]

            if j > 1 and current_val == I[i][j - 1] + gap_extend:
                state = 'I'
            elif j > 1 and current_val == D[i][j - 1] + gap_open:
                state = 'D'
            else:
                state = 'M'

            j -= 1

        elif state == 'D':
            # Вертикальный шаг: гэп в seq2
            aln1 += seq1[i - 1]
            aln2 += '-'

            current_val = D[i][j]

            if i > 1 and current_val == D[i - 1][j] + gap_extend:
                state = 'D'
            elif i > 1 and current_val == I[i - 1][j] + gap_open:
                state = 'I'
            else:
                state = 'M'

            i -= 1

    
    return aln1[::-1], aln2[::-1], score

def print_array(matrix: list):
    for row in matrix:
        for element in row:
            print(f"{element:6}", end="")
        print()

def main():
    aln1, aln2, score = needleman_wunsch_affine("ACGT", "TAGT", gap_open=-10, gap_extend=-1) 
    print(f'str 1: {aln1}')
    print(f'str 2: {aln2}')
    print(f'score: {score}')
    


if __name__ == "__main__":
    main()