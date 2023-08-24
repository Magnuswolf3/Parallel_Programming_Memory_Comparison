/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample demonstrates how use texture fetches in CUDA
 *
 * This sample takes an input PGM image (image_filename) and generates
 * an output PGM image (image_filename_out).  This CUDA kernel performs
 * a simple 2D transform (rotation) on the texture coordinates (u,v).
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define maskSize 3
#define gridSize 16
__constant__ float dMask[maskSize * maskSize];
#define clamp(x) (min(max(x, 0.0), 1.0));

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "tree_bw.pgm";
const char *sampleName = "Global Convolution";

__global__ void gConv(unsigned int width, unsigned int height, float* output, char* imagePath, unsigned int size, float* source)
{
    float sum; 
    //sets up the current row and col index based off the thread positioning
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int maskRad = maskSize/2;


    if(row < height && col < width)
    {
        sum = 0;
        //Changes the starting values of the row and col based off the size of the mask
        int sRow = row - maskRad;
        int sCol = col - maskRad;

        //Iterates through the mask pixel positions
        for(int  i = 0; i < maskSize; i++)
        {
            for(int j = 0; j < maskSize; j++)
            {
                //Gets the index based off its position in the mask
                int currRow = sRow + i;
                int currCol = sCol + j;

                //Checks to see if the pixel position is out of the images bounds
                if(currRow >= 0 && currRow < height && currCol >= 0 && currCol < width)
                {
                    //Sums up the result of the multiplication of the mask and the original image
                    sum += source[(currRow * width + currCol)] * dMask[i * maskSize + j];
                }
                else {
                    sum += 0;
                }
            }
            
        }
        //Sets the value of the current position in the image to the sum of the current positions multiplications
        output[(row * width + col)] = clamp(sum);
    }

}

float globalConvolution(unsigned int width, unsigned int height, float* output, char* imagePath, unsigned int size, float* source)
{
    float time;
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    dim3 dimGrid(ceil((float)width/gridSize), (ceil((float)height/gridSize)));
    dim3 dimBlock(gridSize,gridSize,1);

    gConv<<<dimGrid, dimBlock>>>(width, height, output, imagePath, size, source);

    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);

    gConv<<<dimGrid, dimBlock>>>(width, height, output, imagePath, size, source);
    
    cudaEventRecord(launch_end,0);
    cudaEventSynchronize(launch_end);
    // measure the time spent in the kernel
    time = 0;
    cudaEventElapsedTime(&time, launch_begin, launch_end);

    return time;

}

//Checks to see if the values are out of bounds
bool boundaryWatch( int M,  int x)
{
    if(x < 0)
    {
        return false;
    }
    if(x > M)
    {
        return false;
    }
    return true;
}


double sequentialConvolution(unsigned int width, unsigned int height, float* output, char* imagePath, unsigned int size, float* source,float* mask)
{
    double average_time;
    struct timespec start, end;
    float sum;

    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            sum = 0.0;
            for(int i = -1; i < 2; i++)
            {
                for(int j = -1; j < 2; j++)
                {
                    if(boundaryWatch(width, x+i) && boundaryWatch(height, y+j))
                    {
                        
                        sum += mask[(i+1) +((j+1)*maskSize)]*source[((y+j)*width + (x+i))];
                        
                    }
                }
            }
            output[y*width + x] = clamp(sum);            
        }
    }
    if( clock_gettime( CLOCK_REALTIME, &end) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }
    //compute the time in s
    average_time = (end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e+9;

    return average_time;
}

////////////////////////////////////////////////////////////////////////////////
// Declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("\n----------------------------------------------\n%s starting...\n\n", sampleName);

    runTest(argc, argv);

    cudaDeviceReset();
    printf("\n%s completed", sampleName);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void runTest(int argc, char **argv)
{
    double sequentialTime_1;
    double sequentialTime_2;
    double sequentialTime_3;
    float time_1;
    float time_2;
    float time_3;

    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);


    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);
    
    // Allocate device memory for result
    float *baseOutput_1;
    float *baseOutput_2;
    float *baseOutput_3;
    float *output_1;
    float *output_2;
    float *output_3;
    float *source;


    float sMask[maskSize * maskSize] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    float eMask[maskSize * maskSize] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float aMask[maskSize * maskSize] = {1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0, 1/9.0};

    if(maskSize > 3)
    {
        for(int i = 0; i < maskSize * maskSize; i++)
        {
            aMask[i] = 1/(float)(maskSize * maskSize);
        }
    }

    baseOutput_1 = (float*)malloc(size);
    baseOutput_2 = (float*)malloc(size);
    baseOutput_3 = (float*)malloc(size);
    cudaMalloc((void **) &source, size);
    cudaMalloc((void **) &output_1, size);
    cudaMalloc((void **) &output_2, size);
    cudaMalloc((void **) &output_3, size);

    cudaMemcpy(source, hData, size, cudaMemcpyHostToDevice);
    
    std::cout << "Timing global implementation...\n" << std::endl;

    if(maskSize == 3)
    {sequentialTime_1 = sequentialConvolution(width, height, baseOutput_1, imagePath, size, hData, sMask);
    sequentialTime_2 = sequentialConvolution(width, height, baseOutput_2, imagePath, size, hData, eMask);}
    sequentialTime_3 = sequentialConvolution(width, height, baseOutput_3, imagePath, size, hData, aMask);

    if(maskSize == 3)
    {cudaMemcpyToSymbol(dMask, sMask, maskSize * maskSize * sizeof(float));
    time_1 = globalConvolution(width, height, output_1, imagePath, size, source);
    cudaMemcpyToSymbol(dMask, eMask, maskSize * maskSize * sizeof(float));
    time_2 = globalConvolution(width, height, output_2, imagePath, size, source);}
    cudaMemcpyToSymbol(dMask, aMask, maskSize * maskSize * sizeof(float));
    time_3 = globalConvolution(width, height, output_3, imagePath, size, source);

    //textureConvolution(width, height, output_3, imagePath, size, hData, sMask);

    if(maskSize == 3)
    {printf("Time taken for SharpeningMask:\t%fms\n",time_1);
    printf("Time taken for SobelMask:\t%fms\n",time_2);}
    printf("Time taken for AveragingMask:\t%fms\n\n",time_3);

    if(maskSize == 3){
        printf("Programming speed for SharpeningMask: %.2fMpixels/sec\n", ((width *height) / (((time_1))/1000.0))/1e6);
        printf("Programming speed for SobelMask: %.2fMpixels/sec\n", ((width *height) / (((time_2))/1000.0))/1e6);
        printf("Programming speed for AveragingMask: %.2fMpixels/sec\n\n", ((width *height) / (((time_3))/1000.0))/1e6);
    }
    else {
        printf("Average Programming speed for AveragingMask: %.2fMpixels/sec\n\n", ((width *height) / (((time_3))/1000.0))/1e6);
    }

    if(maskSize == 3)
    {printf("Speedup for Sharpening Mask (SerialTime/GlobalTime): \t%fx\n", ((sequentialTime_1*1000.0)/time_1));
    printf("Speedup for Sobel Mask (SerialTime/GlobalTime): \t%fx\n", ((sequentialTime_2*1000.0)/time_2));}
    printf("Speedup for Averaging Mask (SerialTime/GlobalTime): \t%fx\n\n", ((sequentialTime_3*1000.0)/time_3));
    
    // Allocate mem for the result on host side
    float *hOutputData_1 = (float *) malloc(size);
    float *hOutputData_2 = (float *) malloc(size);
    float *hOutputData_3 = (float *) malloc(size);
    // copy result from device to host
    if(maskSize == 3)
    {cudaMemcpy(hOutputData_1, output_1, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOutputData_2, output_2, size, cudaMemcpyDeviceToHost);}
    cudaMemcpy(hOutputData_3, output_3, size, cudaMemcpyDeviceToHost);

    // Write result to file
    char outputFilename[1024];
    if(maskSize == 3)
    {strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Sharpened_1.pgm");
    sdkSavePGM(outputFilename, baseOutput_1, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_EdgeDetected_1.pgm");
    sdkSavePGM(outputFilename, baseOutput_2, width, height);
    printf("Wrote '%s'\n", outputFilename);}

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Averaged_1.pgm");
    sdkSavePGM(outputFilename, baseOutput_3, width, height);
    printf("Wrote '%s'\n", outputFilename);

    if(maskSize == 3)
    {strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Sharpened_2.pgm");
    sdkSavePGM(outputFilename, hOutputData_1, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_EdgeDetected_2.pgm");
    sdkSavePGM(outputFilename, hOutputData_2, width, height);
    printf("Wrote '%s'\n", outputFilename);}

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Averaged_2.pgm");
    sdkSavePGM(outputFilename, hOutputData_3, width, height);
    printf("Wrote '%s'\n", outputFilename);

    cudaFree(hOutputData_1);
    cudaFree(hOutputData_2);
    cudaFree(hOutputData_3);
    cudaFree(baseOutput_1);
    cudaFree(baseOutput_2);
    cudaFree(baseOutput_3);
    cudaFree(source);
    cudaFree(output_1);
    cudaFree(output_2);
    cudaFree(output_3);
    free(imagePath);
    free(hData);
}
