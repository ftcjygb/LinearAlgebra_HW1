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
    for j in range(cols):
        # Extract column j from M
        col_j = torch.zeros((rows, 1), dtype=torch.float32)
        for i in range(rows):
            col_j[i, 0] = M[i, j]
            
        # Scale the column by the corresponding vector element
        weight = vec[j, 0].item()
        scaled_col = scalar_matrix(weight, col_j)
        
        # Add to the running sum
        vec2 = matrix_sum(vec2, scaled_col)
    
    return vec2