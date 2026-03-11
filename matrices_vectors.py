import torch

def scalar_matrix(s, M):
    rows = M.shape[0]
    cols = M.shape[1]
    # Initialize a tensor of zeros with the same shape as M
    sM = torch.zeros(M.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to scale each entry
    for i in range(rows):
        for j in range(cols):
            sM[i][j] = s * M[i][j]
    return sM

def matrix_sum(M1, M2):
    rows = M1.shape[0]
    cols = M1.shape[1]
    # Initialize a tensor of zeros with the same shape as M1
    M = torch.zeros(M1.shape, dtype=torch.float32)
    # TODO: Complete the functionality by incorporating a for loop to add 
    # corresponding entries of M1 and M2 in a general manner.
    for i in range(rows):
        for j in range(cols):
            M[i][j] = M1[i][j] + M2[i][j]
    return M

def matrix_vector_product(M, vec):
    rows, cols = M.shape[0], M.shape[1]
    vec2 = torch.zeros((rows, 1), dtype=torch.float32)
    
    # TODO: Complete the functionality by implementing a for loop for a general linear combination.
    # Hint: Utilize the scalar_matrix() and matrix_sum() functions.
    
    return vec2
