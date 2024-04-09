#include <iostream>
#include <fstream>
#include <vector>

__global__ void findOddNumbers(const int* A, int* D, int* temp, int numOdd, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size){
        if (temp[tid] == 1){
            int index = 0;
            for(int i = 0; i < tid; i++){
                index += temp[i];
            }
            if(index < numOdd){
                D[index] = A[tid];
            }
        }
    }
}

__global__ void findNumOddNumbers(const int* A, int* d_temp, int* numOdd, int size){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size){
        if (A[tid] % 2 != 0){
            atomicAdd(numOdd, 1);
            d_temp[tid] = 1;
        }
        else {
            d_temp[tid] = 0;
        }
    }
}



int main(int argc, char **argv)
{
    // Implement your solution for question 4. The input file is inp.txt
    // and contains an array A.
    // Running this program should output one file:
    //  (1) q4.txt which contains an array D such that D contains only the odd
    //      numbers from the input array. You should preserve the order of the
    //      numbers as they are in the input array.

    // Read input array from inp.txt
    std::ifstream inputFile("inp.txt");
    if (!inputFile){
        std::cerr << "Failed to open inp.txt" << std::endl;
        return 1;
    }

    std::vector<int> A;
    std::string line;
    while (std::getline(inputFile, line, ',')){
        int value = std::stoi(line);
        A.push_back(value);
    }
    inputFile.close();
    int size = A.size();

    int* d_A;
    int* d_temp;
    int* d_numOdd;
    int numOdd = 0;
    
    // Allocate memory on the GPU
    cudaMalloc((void**)&d_A, size * sizeof(int));
    cudaMalloc((void**)&d_temp, size * sizeof(int));
    cudaMalloc((void**)&d_numOdd, sizeof(int));

    // Copy input array from host to device
    cudaMemcpy(d_A, A.data(), size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_numOdd, &numOdd, sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel to find number of odd numbers
    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;
    findNumOddNumbers<<<gridSize, blockSize>>>(d_A, d_temp, d_numOdd, size);

    // Copy number of odd numbers from device to host
    cudaMemcpy(&numOdd, d_numOdd, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Number of odd numbers: %d\n", numOdd);

    // Allocate memory for array D on the GPU
    int* d_D;
    cudaMalloc((void**)&d_D, numOdd * sizeof(int));

    // Launch kernel to find and copy odd numbers to array D
    findOddNumbers<<<gridSize, blockSize>>>(d_A, d_D, d_temp, numOdd, size);

    // Copy array D from device to host
    std::vector<int> D(numOdd);
    cudaMemcpy(D.data(), d_D, numOdd * sizeof(int), cudaMemcpyDeviceToHost);

    // Write array D to q4.txt
    std::ofstream outputFile("q4.txt");
    if (!outputFile){
        std::cerr << "Failed to open q4.txt" << std::endl;
        return 1;
    }

    for (int i = 0; i < numOdd; i++){
        if(i == numOdd - 1){
            outputFile << D[i];
        }
        else{
            outputFile << D[i] << ", ";
        }
    }
    outputFile.close();

    // Free memory
    cudaFree(d_A);
    cudaFree(d_temp);
    cudaFree(d_numOdd);
    cudaFree(d_D);

    return 0;
}
