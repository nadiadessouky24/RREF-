import numpy as np

def REF(matrix):
    matrix = matrix.copy()
    row_size, col_size = matrix.shape

    def type1_ERO(row1, row2):
        matrix[[row1, row2]] = matrix[[row2, row1]]
    
    def type2_ERO(row, col):
        matrix[row] = matrix[row] / matrix[row, col]

    def type3_ERO(row1, row2, col):
        factor = matrix[row1, col]
        matrix[row1] = matrix[row1] - factor * matrix[row2]
    
    r = 0
    for c in range(col_size):
        if r >= row_size:
            break
        
        pivot_row = None
        for i in range(r, row_size):
            if matrix[i, c] != 0:
                pivot_row = i
                break
        
        if pivot_row is None:
            continue
        
        if pivot_row != r:
            type1_ERO(r, pivot_row)
        
        type2_ERO(r, c)
        
        for i in range(r + 1, row_size):
            type3_ERO(i, r, c)
        
        r += 1
    
    return matrix

def RREF(matrix):
    matrix = REF(matrix)
    row_size, col_size = matrix.shape

    for r in range(row_size - 1, -1, -1):
        lead_col = None
        for c in range(col_size):
            if matrix[r, c] == 1:
                lead_col = c
                break
        if lead_col is None:
            continue
        for i in range(r):
            factor = matrix[i, lead_col]
            matrix[i] = matrix[i] - factor * matrix[r]
    
    return matrix

def inverse_and_determinant(matrix):
    matrix = np.array(matrix, dtype=float)
    rows, cols = matrix.shape

    if rows != cols:
        return "Matrix is not square", None

    augmented_matrix = np.hstack((matrix, np.eye(rows)))
    rref_matrix = RREF(augmented_matrix)

    rref_only_matrix = rref_matrix[:, :cols]
    identity_matrix = np.eye(rows)

    if not np.allclose(rref_only_matrix, identity_matrix):
        return "Matrix is not invertible", None

    inverse_matrix = rref_matrix[:, cols:]

    # determinant = np.linalg.det(matrix)
    n = len(matrix)

    if n == 1:
        determinant =  matrix[0][0]

    if n == 2:
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    A = [row[:] for row in matrix]

    for i in range(n):
        max_el = abs(A[i][i])
        max_row = i
        for k in range(i+1, n):
            if abs(A[k][i]) > max_el:
                max_el = abs(A[k][i])
                max_row = k

        A[i], A[max_row] = A[max_row], A[i]

        for k in range(i+1, n):
            c = -A[k][i] / A[i][i]
            for j in range(i, n):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += c * A[i][j]

    determinant = 1
    for i in range(n):
        determinant *= A[i][i]

    return inverse_matrix, determinant

matrix = np.array([
    [2, 1, 1, 0],
    [4, -6, 0, -2],
    [-2, 7, 2, 3],
    [4, 2, -5, 1]
])

rref_matrix = RREF(matrix)
print("RREF Matrix:")
print(rref_matrix)

inverse_det = inverse_and_determinant(matrix)

if isinstance(inverse_det, str):
    print(inverse_det)
else:
    inverse_matrix, determinant = inverse_det
    print("Inverse Matrix:")
    print(inverse_matrix)
    print("Determinant:")
    print(determinant)
