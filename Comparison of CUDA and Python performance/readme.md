## Problem:
Write CUDA C based codes to conduct the following multiple matrix mulipli-
cation operations. Given N matrixes A1; ... ;AN, and N matrix muliplication
B1; ... ;BN, the N matrixes C1; ... ;CN could be obtained by the following
equations.</br>
C1 =A1 * B1</br>
C2 =A2 * B2</br>
...</br>
CN =AN * BN</br>
#### Task-1:where A1; ... ;AN and B1; ... ;BN have the same shape, and the matrixes A and B are randomly generated.</br>
#### Task-2:where A1; ... ;AN and B1; ... ;BN have the same shape, and the matrixes A is randomly generated and B is static.</br>
• Students MUST use CUDA C to conduct the operation. Other coding environments/libraries such as PyCuda, MATLAB, CuPy are NOT allow to use.</br>
However, students could use cuBLAS library to complete the assignment as cuBLAS is one of the most important libraries in CUDA.</br>
• Please run your codes under the following ONE of the three conditions.</br>
i)A = R500 * 500, B = R500 * 400;N > 100; </br>
ii) A = R50 * 20, B = R20 * 50;N > 5000;</br>
iii) A = R6 * 4000, B = R4000 * 9;N > 1000;</br>

## Solution:
#### Task-1:
Here Matrix A is 5000 * 5000 and Matrix B is 5000 * 4000 and number of iterations are 200.</br>
• Matrix A and B are randomly generated for both cases.</br>
• For python implementation, I have used matmul, Numpy function to implement the matrix multiplication.</br>
• For Cuda, I have used cuBLAS and cuRAND library to implement the matrix multiplication.</br>
|  | Performance of CPU| Performance of GPU|
| ----------- | ----------- | ----------- |
|Condition-1| 336369.8 ms| 13056.52 ms|
#### Task-2:
For python implementation, I have used matmul, Numpy function to implement the matrix multiplication.</br>
• For Cuda, I have used cuBLAS and cuRAND library to implement the matrix multiplication.</br>
• In both cases, B is static for each iteration and A is randomly generated.</br>

|   | Performance of CPU | Performance of GPU|
| ----------- | ----------- | ----------- |
|Condition-1| 12979.4 ms| 720.81 ms|
|Condition-2| 278.3 ms |1923.19 ms|
|Condition-3| 589.4 ms |878.82 ms|

