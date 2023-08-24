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

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define maskSize 3
#define MAX_EPSILON_ERROR 5e-3f
#define clamp(x) (min(max(x, 0.0), 1.0));

// Define the files that are to be save and the reference images for validation
const char *imageFilename = "tree_bw.pgm";

const char *sampleName = "Sequential Convolution";


//Checks to see if the values are out of bounds
bool boundaryWatch( int M,  int x)
{
    //If position checked is lower than zero return false
    if(x < 0)
    {
        return false;
    }

    //If position checked is greater than size of image, return false
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

    //Start calculating time taken for this process
    if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) {
        perror( "clock gettime" );
        exit( EXIT_FAILURE );
    }

    //Iterate through all image pixels
    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
        {
            //Initialise sum for current mask pixel
            sum = 0.0;
            
            //Iterate through all mask pixel
            for(int i = -1; i < 2; i++)
            {
                for(int j = -1; j < 2; j++)
                {
                    if(boundaryWatch(width, x+i) && boundaryWatch(height, y+j))
                    {
                        //Sum up all the mask multiplication values for current mask pixel
                        sum += mask[(i+1) +((j+1)*maskSize)]*source[((y+j)*width + (x+i))];
                    }
                }
            }

            //Sets the value of the current pixel position to the convolution result
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

    baseOutput_1 = (float*)malloc(size);
    baseOutput_2 = (float*)malloc(size);
    baseOutput_3 = (float*)malloc(size);

    // Setup masks
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

    std::cout << "Timing sequential implementation...\n" << std::endl;

    if(maskSize == 3)
    {sequentialTime_1 = sequentialConvolution(width, height, baseOutput_1, imagePath, size, hData, sMask);
    sequentialTime_2 = sequentialConvolution(width, height, baseOutput_2, imagePath, size, hData, eMask);}
    sequentialTime_3 = sequentialConvolution(width, height, baseOutput_3, imagePath, size, hData, aMask);

    if(maskSize == 3)
    {printf("Time taken for SharpeningMask:\t%fms\n",sequentialTime_1*1000.0);
    printf("Time taken for SobelMask:\t%fms\n",sequentialTime_2*1000.0);}
    printf("Time taken for AveragingMask:\t%fms\n\n",sequentialTime_3*1000.0);
    if(maskSize == 3)
    {printf("Programming speed for SharpeningMask: %.2fMpixels/sec\n", ((width *height) / ((sequentialTime_1)))/1e6);
    printf("Programming speed for SobelMask: %.2fMpixels/sec\n", ((width *height) / ((sequentialTime_2)))/1e6);}
    printf("Programming speed for AveragingMask: %.2fMpixels/sec\n\n", ((width *height) / ((sequentialTime_3)))/1e6);

    // Write results to their respective files
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

    cudaFree(baseOutput_1);
    cudaFree(baseOutput_2);
    cudaFree(baseOutput_3);
    free(imagePath);
}
