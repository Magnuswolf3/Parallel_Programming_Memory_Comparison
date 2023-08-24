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

#define TILE_WIDTH 16
#define maskSize 3
#define memElements (TILE_WIDTH + maskSize - 1)
#define clamp(x) (min(max(x, 0.0), 1.0));

__constant__ float dMask[maskSize * maskSize];

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "tree_bw.pgm";
const char *sampleName = "Shared Convolution";

__global__ void sConv(unsigned int width, unsigned int height, float* output, char* imagePath, unsigned int size, float* source)
{
    //Sets up the shared memory space to use for storing the image
    __shared__ float sharedImage[memElements][memElements];

    int maskRad = maskSize/2;
    float sum; 

    //Indexing for the shared memory space's rows and columns 
    int d = threadIdx.y * TILE_WIDTH + threadIdx.x;
    int dx = d%memElements;
    int dy = d/memElements;

    //Indexing for the original images row's and columns
    int sx = blockIdx.x * TILE_WIDTH + dx - maskSize;
    int sy = blockIdx.y * TILE_WIDTH + dy - maskSize;
    int s = (sy * width) + sx;

    //Checks to see if current position is out of bounds for the image
    if(sx >= 0 && sx < width && sy >= 0 && sy < height)
    {
        //Sets the value of shared memory pixel to value of original image pixel
        sharedImage[dy][dx] = source[s];
    }
    else{
        sharedImage[dy][dx] = 0;
    }


    //Fills in the rest of the shared memory spaces for this block
    for(int i = 1; i <= (memElements*memElements)/(TILE_WIDTH * TILE_WIDTH); i++)
    {
        d = threadIdx.y * TILE_WIDTH + threadIdx.x + i * TILE_WIDTH * TILE_WIDTH;
        dx = d%memElements;
        dy = d/memElements;

        sx = blockIdx.x * TILE_WIDTH + dx - 3 * maskRad;
        sy = blockIdx.y * TILE_WIDTH + dy - 3 * maskRad;

        if(dy < memElements)
        {     
            if(sx >= 0 && sx < width && sy >= 0 && sy < height)
            {
                //Fills in the next few pixels of the source images for this current thread block
                sharedImage[dy][dx] = source[sy * width + sx];
            }
            else
            {
                sharedImage[dy][dx] = 0;
            }
        }
    }
     
    //makes sure all threads have been completed
    __syncthreads();
    sum = 0;
    for(int i = 0; i < maskSize; i++)
    {
        for(int j = 0; j < maskSize; j++)
        {
            //performs convolution on current thread position using the mask in constant memory and the image in shared memory space
            sum += sharedImage[threadIdx.y +i][threadIdx.x + j] * dMask[i * maskSize + j];
        }
    }

    //sets the threadIndex for current pixel position
    int x = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int y = blockIdx.y * TILE_WIDTH + threadIdx.y;

    if(y < height && x < width)
    {
        //sets value to convolution result for current mask position
        output[(y * width) + x] = clamp(sum);
    }
    //makes sure all the threads have been completed by this point
    __syncthreads();
}

float sharedConvolution(unsigned int width, unsigned int height, float* output, char* imagePath, unsigned int size, float* source)
{
    float time;
    cudaEvent_t launch_begin, launch_end;
    cudaEventCreate(&launch_begin);
    cudaEventCreate(&launch_end);

    dim3 dimGrid(ceil((float)width/TILE_WIDTH), (ceil((float)height/TILE_WIDTH)));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH,1);

    

    sConv<<<dimGrid, dimBlock>>>(width, height, output, imagePath, size, source);

    // record a CUDA event immediately before and after the kernel launch
    cudaEventRecord(launch_begin,0);

    sConv<<<dimGrid, dimBlock>>>(width, height, output, imagePath, size, source);

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
    //cudaMalloc((void **) &dMask , maskSize * maskSize * sizeof(float));

    cudaMemcpy(source, hData, size, cudaMemcpyHostToDevice);
    
    std::cout << "Timing global implementation...\n" << std::endl;

    if(maskSize == 3){
    sequentialTime_1 = sequentialConvolution(width, height, baseOutput_1, imagePath, size, hData, sMask);
    sequentialTime_2 = sequentialConvolution(width, height, baseOutput_2, imagePath, size, hData, eMask);
    }
    sequentialTime_3 = sequentialConvolution(width, height, baseOutput_3, imagePath, size, hData, aMask);

    if(maskSize == 3)
    {cudaMemcpyToSymbol(dMask, sMask, maskSize * maskSize * sizeof(float));
    time_1 = sharedConvolution(width, height, output_1, imagePath, size, source);

    cudaMemcpyToSymbol(dMask, eMask, maskSize * maskSize * sizeof(float));
    time_2 = sharedConvolution(width, height, output_2, imagePath, size, source);}

    cudaMemcpyToSymbol(dMask, aMask, maskSize * maskSize * sizeof(float));
    time_3 = sharedConvolution(width, height, output_3, imagePath, size, source);

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
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Sharpened_3.pgm");
    sdkSavePGM(outputFilename, hOutputData_1, width, height);
    printf("Wrote '%s'\n", outputFilename);

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_EdgeDetected_3.pgm");
    sdkSavePGM(outputFilename, hOutputData_2, width, height);
    printf("Wrote '%s'\n", outputFilename);}

    strcpy(outputFilename, imagePath);
    strcpy(outputFilename + strlen(imagePath) - 4, "_out_Averaged_3.pgm");
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
