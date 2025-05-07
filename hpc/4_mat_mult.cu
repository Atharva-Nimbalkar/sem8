#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matrixMul(int* a, int* b, int* c, int rowsA, int colsA, int colsB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB) {
        int sum = 0;
        for (int i = 0; i < colsA; i++) {
            sum += a[row * colsA + i] * b[i * colsB + col];
        }
        c[row * colsB + col] = sum;
    }
}

int main() {
    int rowsA, colsA, rowsB, colsB;

    cout << "Enter rows and columns for Matrix A: ";
    cin >> rowsA >> colsA;

    cout << "Enter rows and columns for Matrix B: ";
    cin >> rowsB >> colsB;

    if (colsA != rowsB) {
        cout << "Matrix multiplication not possible: colsA != rowsB" << endl;
        return -1;
    }

    // Host vectors
    vector<int> a(rowsA * colsA), b(rowsB * colsB), c(rowsA * colsB);

    // Input Matrix A
    cout << "Enter elements of Matrix A:" << endl;
    for (int i = 0; i < rowsA * colsA; ++i) {
        cin >> a[i];
    }

    // Input Matrix B
    cout << "Enter elements of Matrix B:" << endl;
    for (int i = 0; i < rowsB * colsB; ++i) {
        cin >> b[i];
    }

    // Device pointers
    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void**)&dev_a, rowsA * colsA * sizeof(int));
    cudaMalloc((void**)&dev_b, rowsB * colsB * sizeof(int));
    cudaMalloc((void**)&dev_c, rowsA * colsB * sizeof(int));

    cudaMemcpy(dev_a, a.data(), rowsA * colsA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), rowsB * colsB * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + 15) / 16, (rowsA + 15) / 16);
    matrixMul<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, rowsA, colsA, colsB);

    // Copy result back
    cudaMemcpy(c.data(), dev_c, rowsA * colsB * sizeof(int), cudaMemcpyDeviceToHost);

    // Output Result
    cout << "Result Matrix:" << endl;
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            cout << c[i * colsB + j] << " ";
        }
        cout << endl;
    }

    // Free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}