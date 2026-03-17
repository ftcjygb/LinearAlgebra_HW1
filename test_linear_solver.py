import torch
from linear_solver import solve_linear_equations
from matrices_vectors import matrix_vector_product, matrix_sum, scalar_matrix

def run_test(A, b, test_name):
    print(f"--- Testing: {test_name} ---")
    x = solve_linear_equations(A, b)
    
    if x is None:
        print("Result: No Solution (Inconsistent)\n")
    else:
        print("Result: Solution Vector x:")
        print(x)
        
        # TODO: Calculate Error = Ax - b (it should be close to zero)
        # Hint: Use our custom matrix_vector_product to compute the linear combination: Ax
        # Hint: Use our custom scalar_matrix and matrix_sum to get the error vector: Ax-b
        # Hint: Get the L2 Norm (Euclidean distance) of the error vector (you may use torch.norm)
        # Hint: Report the L2 Norm result
        Ax = matrix_vector_product(A, x)
        neg_b = scalar_matrix(-1.0, b)
        error_vec = matrix_sum(Ax, neg_b)
        
        error_norm = torch.norm(error_vec).item()
        print(f"L2 Norm of Error (||Ax - b||): {error_norm:.6e}\n")


# 1. Define Test Data (4 equations, 4 unknowns)
# Case 1: Consistent System (Unique solution or Infinite)
A_consistent = torch.tensor([
    [1, 2, 1, -1],
    [3, 2, 4,  4],
    [4, 4, 3,  4],
    [2, 0, 1,  5]
], dtype=torch.float32)

b_consistent = torch.tensor([[5], [16], [22], [15]], dtype=torch.float32)

# Case 2: Inconsistent System
# (Row 3 is a multiple of Row 1, but b is not)
A_inconsistent = torch.tensor([
    [1, 1, 1, 1],
    [2, 3, 1, 4],
    [1, 1, 1, 1], # Same coefficients as row 1
    [0, 1, 2, 3]
], dtype=torch.float32)

b_inconsistent = torch.tensor([[10], [20], [5], [15]], dtype=torch.float32) # b[2] contradicts b[0]

# BONUS: System Robustness & Scalability Challenge
# Consider to define the following cases and report the results.
#
# 1. Minimal Case (2x2 or 3x3):
#    - Test a simple, manually verifiable system to ensure core logic is sound.
#
# 2. Non-Square (Rectangular) Matrices:
#    - Underdetermined: "Fat" matrix (e.g., 3x5). Does it handle free variables?
#    - Overdetermined: "Tall" matrix (e.g., 6x3). Test for both consistent 
#      and inconsistent cases.
#
# 3. Large Scale (100x100+):
#    - Generate a large matrix A and vector x_true. 
#
# 4. Report Requirements:
#    - List the dimensions and properties of your test data.
#    - Provide the final verification error ||Ax - b|| for each case.

A_minimal = torch.tensor([
    [1, 2],
    [4, 5]
], dtype=torch.float32)
b_minimal = torch.tensor([[3], [6]], dtype=torch.float32)

A_under = torch.tensor([
    [1, 2, 3, 4, 5],
    [2, 5, 1, 3, 2],
    [3, 7, 4, 7, 7]
], dtype=torch.float32)
b_under = torch.tensor([[10], [15], [25]], dtype=torch.float32)

A_over_c = torch.tensor([
    [1, 1, 1],
    [2, 1, -1],
    [3, 2, 0],
    [1, 0, -2],
    [4, 3, 1]
], dtype=torch.float32)
b_over_c = torch.tensor([[6], [1], [7], [-5], [13]], dtype=torch.float32)

b_over_i = torch.tensor([[6], [1], [7], [-5], [99]], dtype=torch.float32)


A_large = torch.randn((100, 100), dtype=torch.float32)
x_true_large = torch.ones((100, 1), dtype=torch.float32)
b_large = torch.matmul(A_large, x_true_large)

if __name__ == "__main__":
    run_test(A_consistent, b_consistent, "Consistent Case")
    run_test(A_inconsistent, b_inconsistent, "Inconsistent Case")

    print("====== BONUS: Robustness & Scalability Tests ======\n")
    run_test(A_minimal, b_minimal, "Minimal 3x3 Case")
    run_test(A_under, b_under, "Underdetermined 3x5 (Fat Matrix)")
    run_test(A_over_c, b_over_c, "Overdetermined 5x3 Consistent (Tall Matrix)")
    run_test(A_over_c, b_over_i, "Overdetermined 5x3 Inconsistent (Tall Matrix)")
    run_test(A_large, b_large, "Large Scale 100x100 Matrix")