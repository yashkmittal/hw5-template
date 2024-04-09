#include <iostream>
#include <fstream>
#include <vector>

__global__ void findMinValue(const int* A, int size, int* minVal)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        atomicMin(minVal, A[tid]);
    }
}

__global__ void extractLastDigit(const int* A, int* B, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        B[tid] = A[tid] % 10;
    }
}

int main(int argc, char **argv)
{
    // Read input array from inp.txt
    std::ifstream inputFile("inp.txt");
    if (!inputFile)
    {
        std::cerr << "Failed to open inp.txt" << std::endl;
        return 1;
    }

    std::vector<int> A;
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
    int* d_minVal;

    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, size * sizeof(int));
    cudaMalloc((void**)&d_B, size * sizeof(int));
    cudaMalloc((void**)&d_minVal, sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_A, A.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    // Set initial minimum value to maximum possible value
    int minVal = INT_MAX;
    cudaMemcpy(d_minVal, &minVal, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to find minimum value
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    findMinValue<<<gridSize, blockSize>>>(d_A, size, d_minVal);

    // Copy minimum value from device to host
    cudaMemcpy(&minVal, d_minVal, sizeof(int), cudaMemcpyDeviceToHost);

    // Launch kernel to extract last digit
    extractLastDigit<<<gridSize, blockSize>>>(d_A, d_B, size);

    // Copy output array from device to host
    std::vector<int> B(size);
    cudaMemcpy(B.data(), d_B, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Write minimum value to q2a.txt
    std::ofstream outputFileA("q2a.txt");
    if (!outputFileA)
    {
        std::cerr << "Failed to open q2a.txt" << std::endl;
        return 1;
    }
    outputFileA << minVal;
    outputFileA.close();

    // Write output array B to q2b.txt
    std::ofstream outputFileB("q2b.txt");
    if (!outputFileB)
    {
        std::cerr << "Failed to open q2b.txt" << std::endl;
        return 1;
    }
    for (int i = 0; i < size; i++)
    {
        outputFileB << B[i] << " ";
    }
    outputFileB.close();

    // Free memory on the GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_minVal);

    return 0;
}
