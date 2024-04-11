#include <iostream>
#include <fstream>
#include <vector>

__global__ void findFrequency(const int* A, int* B, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size){
        atomicAdd(&B[A[tid]/100], 1);
    }
}

__global__ void findLocalFrequency(const int* A, int* B, int size) {
    __shared__ int localB[10]; // Shared memory for local copy of B within each block
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x < 10)
        localB[threadIdx.x] = 0;

    __syncthreads();

    if (tid < size) {
        atomicAdd(&localB[A[tid] / 100], 1);
    }

    __syncthreads();

    // Add localB to global B
    if (threadIdx.x < 10)
        atomicAdd(&B[threadIdx.x], localB[threadIdx.x]);
}

__global__ void computePrefix(int* B, int* C) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    int result = 0;

    for (int i = 0; i <= tid; i++) {
        result += B[i];
    }

    C[tid] = result;
}

int main(int argc, char **argv)
{
    // Implement your solution for question 3. The input file is inp.txt
    // and contains an array A (range of values is 0-999).
    // Running this program should output three files:
    //  (1) q3a.txt which contains an array B of size 10 that keeps a count of
    //      the entries in each of the ranges: [0, 99], [100, 199], [200, 299], ..., [900, 999].
    //      For this part, the array B should reside in global GPU memory during computation.
    //  (2) q3b.txt which contains the same array B as in the previous part. However,
    //      you must use shared memory to represent a local copy of B in each block, and
    //      combine all local copies at the end to get a global copy of B.
    //  (3) q3c.txt which contains an array C of size 10 that keeps a count of
    //      the entries in each of the ranges: [0, 99], [0, 199], [0, 299], ..., [0, 999].
    //      You should only use array B for this part (do not use the original input array A).

    // Read input array from inp.txt
    std::ifstream inputFile("inp.txt");
    if (!inputFile)
    {
        std::cerr << "Failed to open inp.txt" << std::endl;
        return 1;
    }

    std::vector<int> A;
    std::vector<int> B(10, 0); // Initialize B to 0 (10 elements)
    std::vector<int> C(10, 0);

    std::string line;
    while (std::getline(inputFile, line, ','))
    {
        int value = std::stoi(line);
        A.push_back(value);
    }
    inputFile.close();

    int size = A.size();
    int* d_A;
    int* d_B;
    int* d_C;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, size * sizeof(int));
    cudaMalloc((void**)&d_B, 10 * sizeof(int));
    cudaMalloc((void**)&d_C, 10 * sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_A, A.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), 10 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel to find frequency
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    findFrequency<<<gridSize, blockSize>>>(d_A, d_B, size);

    // Copy result back to host
    cudaMemcpy(B.data(), d_B, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output to q3a.txt
    std::ofstream outputFileA("q3a.txt");
    if (!outputFileA)
    {
        std::cerr << "Failed to open q3a.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < 10; i++)
    {
        if (i != 9) {
            outputFileA << B[i] << ", ";
        } else {
            outputFileA << B[i];
        }
    }
    outputFileA.close();

    // ------
    // Part B
    // ------

    cudaMemset(d_B, 0, 10 * sizeof(int)); // Reset d_B to 0

    findLocalFrequency<<<gridSize, blockSize>>>(d_A, d_B, size);

    // Copy result back to host
    cudaMemcpy(B.data(), d_B, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output to q3b.txt
    std::ofstream outputFileB("q3b.txt");
    if (!outputFileB)
    {
        std::cerr << "Failed to open q3b.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < 10; i++)
    {
        if (i != 9) {
            outputFileB << B[i] << ", ";
        } else {
            outputFileB << B[i];
        }
    }
    outputFileB.close();

    // ------
    // Part C
    // ------

    computePrefix<<<1, 10>>>(d_B, d_C);
    cudaMemcpy(C.data(), d_C, 10 * sizeof(int), cudaMemcpyDeviceToHost);

    // Write output to q3c.txt
    std::ofstream outputFileC("q3c.txt");
    if (!outputFileC)
    {
        std::cerr << "Failed to open q3c.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < 10; i++)
    {
        if (i != 9) {
            outputFileC << C[i] << ", ";
        } else {
            outputFileC << C[i];
        }
    }
    outputFileC.close();

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
