import torch
from matrices_vectors import matrix_vector_product
from gauss import gauss_elimination

def test_consistency(augmented_RREF):
    """
    Checks if the system Ax = b is consistent.
    """
    rows, cols = augmented_RREF.shape
    # TODO: Use the following property to determine consistency:
    # A system is inconsistent if and only if a row in the RREF looks like [0 0 ... 0 | b_i] 
    # where b_i != 0. 
    # Return False if inconsistent 
    zero_thresh = 1e-5
    for i in range(rows):
        is_zero_row = True
        for j in range(cols - 1):
            if torch.abs(augmented_RREF[i, j]) > zero_thresh:
                is_zero_row = False
                break
        if is_zero_row and torch.abs(augmented_RREF[i, cols - 1]) > zero_thresh:
            return False

    return True # Consistent

def generate_solution(augmented_RREF):
    """
    Extracts a solution vector. Basic variables are solved, Free variables are set to 1.
    """
    rows, cols_aug = augmented_RREF.shape
    num_vars = cols_aug - 1
    solution = torch.ones((num_vars, 1), dtype=torch.float32) # Default free variables to 1
    
    # TODO: Identify pivot columns
    # Hint: Find the first nonzero in each row
    zero_thresh = 1e-5
    
    # TODO: Calculate basic variables
    # Hint: Note that we have assume all free variables are set to 1
    for i in range(rows):
        for j in range(num_vars):
            if torch.abs(augmented_RREF[i, j]) > zero_thresh:
                # Pivot found at column j
                b_val = augmented_RREF[i, -1].item()
                # Subtract the contribution of the free variables (which are 1)
                for k in range(j + 1, num_vars):
                    b_val -= augmented_RREF[i, k].item() * 1.0
                solution[j, 0] = b_val
                break
            
    return solution

def solve_linear_equations(A, b):
    """
    Wrapper function to solve Ax = b.
    """

    # Create Augmented Matrix [A | b]
    augmented_matrix = torch.cat((A, b), dim=1)

    rows, cols_aug = augmented_matrix.shape
    num_vars = cols_aug - 1
    solution = torch.zeros((num_vars, 1), dtype=torch.float32) # Solution is default to zero
    
    # TODO: Use our custom gauss_elimination to obtain the Reduced Row Echelon Form.
    # Then, use test_consistency to inspect the reduced matrix.
    # Return None if the system is inconsistent. Otherwise, return the results of generate_solution.
    rref_matrix = gauss_elimination(augmented_matrix)
    
    if not test_consistency(rref_matrix):
        return None
        
    solution = generate_solution(rref_matrix)
    return solution