#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for vector addition
__global__ void vectorAdd(int* a, int* b, int* c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int size;
    cout << "Enter the size of the vectors: ";
    cin >> size;

    vector<int> a(size), b(size), c(size);

    cout << "Enter elements of vector A:\n";
    for (int i = 0; i < size; i++) {
        cin >> a[i];
    }

    cout << "Enter elements of vector B:\n";
    for (int i = 0; i < size; i++) {
        cin >> b[i];
    }

    int* dev_a, * dev_b, * dev_c;
    cudaMalloc((void**)&dev_a, size * sizeof(int));
    cudaMalloc((void**)&dev_b, size * sizeof(int));
    cudaMalloc((void**)&dev_c, size * sizeof(int));

    cudaMemcpy(dev_a, a.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int gridSize = (size + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(dev_a, dev_b, dev_c, size);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "CUDA error: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    cudaMemcpy(c.data(), dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    cout << "Result of vector addition:\n";
    for (int i = 0; i < size; i++) {
        cout << a[i] << " + " << b[i] << " = " << c[i] << endl;
    }

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}