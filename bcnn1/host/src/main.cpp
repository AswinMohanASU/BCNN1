//
// Created by Aswin Gunavelu Mohan on 7/14/16.
//

/**
* This implements BCNN
*  on the "Altera" FPGA using OpenCL 1.0.
*
* @author  Aswin Gunavelu Mohan <gmaswin@gmail.com>
* @version 1.0
* @since   2016-06-08
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <iostream>
#include "CL/opencl.h"
#include "AOCL_Utils.h"
#include "Utility.h"
#include "data.h"
using namespace std;
using namespace aocl_utils;
#define AOCL_ALIGNMENT 64
#define N 1
unsigned int correct;
cl_int err;
cl_uint numPlatforms;

//int *X1 = (int*) memalign ( AOCL_ALIGNMENT, (sizeof(int)));

size_t global[3];                       // global domain size for our calculation
size_t local[3];                       	// local domain size for our calculation

cl_platform_id platform;                // compute platform id
cl_device_id device;                	// compute device id
cl_context context;                 	// compute context

cl_command_queue queue[N];             // compute command queue
cl_program program;                	    // compute program
cl_kernel kernel[N];                   	// compute kernel


int h_debug[3];
int h_offset[N];
int i,j;

scoped_array<cl_mem> d_fmap0;
scoped_array<cl_mem> d_norm1;
scoped_array<cl_mem> d_w1;
scoped_array<cl_mem> d_debug;
scoped_array<cl_mem> d_offset;
cl_mem d_fmap1;
cl_mem d_act1;

scoped_aligned_ptr<int> h_fmap0;
scoped_aligned_ptr<int> h_w1;
scoped_aligned_ptr<int> h_norm1;
scoped_aligned_ptr<int> h_fmap1;
scoped_aligned_ptr<int> h_act1;

void cleanup();


int main(void){

    h_fmap0.reset(3*34*34);
    for(i = 0; i < 3*34*34; ++i) {
        h_fmap0[i] = fmap0[i];
    }
    h_w1.reset(128*3*3*3);
    for(i = 0 ; i < 128*3*3*3 ; i ++){
        h_w1[i] = w1[i];
    }
    h_norm1.reset(128);
    for(i = 0 ; i < 128 ; i ++){
        h_norm1[i] = norm1[i];
    }
    h_fmap1.reset(128*34*34);
    h_act1.reset(128*32*32);

    d_fmap0.reset(N);
    d_norm1.reset(N);
    d_w1.reset(N);
    d_debug.reset(N);
    d_offset.reset(N);
    char* Plt = "Altera";

    //Get PlatformID
    platform = find_Platform(Plt);

    //Get DeviceID
    //
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 1, &device, NULL);
    checkError(err,"Error: Failed to get DeviceID!");
    // Create a compute context
    //
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    checkError(err,"Error: Failed to Create context!");

    string binary_file = getBoardBinaryFile("kernel_e1", device);
    printf("Using AOCX: %s\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    if (!program)
    {
        printf("Error: Failed to create compute program!\n");
        return EXIT_FAILURE;

    }

    // Build the program executable
    //
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS)
    {
        size_t len;
        char buffer[4096];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
        exit(1);
    }

    // Create the input and output arrays in device memory for our calculation
    //

    for(i = 0; i < N ; i ++){
    d_fmap0[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 3 * 34 * 34, NULL, NULL);
    d_w1[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 128 * 3 * 3 * 3, NULL, NULL);
    d_norm1[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int) * 128, NULL, NULL);
    d_offset[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int), NULL, NULL);
    d_debug[i] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*3, NULL, NULL);
    }
    d_fmap1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 128 * 34 * 34, NULL, NULL);
    d_act1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 128 * 32 * 32, NULL, NULL);

    h_debug = {128,32,32};
    printf("Completed Buffer Creation \n");
    cl_event event_kernel[N];

    global = {1, 1, 1};
    h_offset[0] = 0;
        for(i = 1; i < N ; i ++)
     		h_offset[i] = h_offset[i-1] + 8;
     	

for(i = 0; i < N ; i ++){
        queue[i] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Error: Failed to create a command queue[%d]!",i);  
        
        // Create the compute kernel in the program we wish to run
        //
        kernel[i] = clCreateKernel(program, "conv", &err);
        checkError(err, "Error: Failed to create compute kernel[%d]!",i);

        // Write our data set into the input array in device memory
        //
        err = clEnqueueWriteBuffer(queue[i], d_fmap0[i], CL_FALSE, 0, sizeof(int) * 3 * 34 * 34, h_fmap0, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_fmap0",i);

        err = clEnqueueWriteBuffer(queue[i], d_w1[i], CL_FALSE, 0, sizeof(int) * 128 * 3 * 3 * 3, h_w1, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w1",i);

        err = clEnqueueWriteBuffer(queue[i], d_norm1[i], CL_FALSE, 0, sizeof(int) * 128, h_norm1, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm1",i);

        err = clEnqueueWriteBuffer(queue[i], d_debug[i], CL_FALSE, 0, sizeof(int) * 3, h_debug, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_debug",i);	

        err = clEnqueueWriteBuffer(queue[i], d_offset[i], CL_FALSE, 0, sizeof(int), &h_offset[i], 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_offset",i);
       

        // Set the arguments to our compute kernel
        //
        unsigned argi = 0;
        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_fmap0[i]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap0",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_w1[i]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w1",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_norm1[i]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm1",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_debug[i]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_debug",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_offset[i]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_offset",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_act1);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act1",i);

        err = clSetKernelArg(kernel[i], argi++, sizeof(cl_mem), &d_fmap1);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act1",i);

    }
    printf("Completed Setting Arguments \n");
            err = clEnqueueNDRangeKernel(queue[0], kernel[0], 3, NULL, global, NULL, 0, NULL, NULL);
            checkError(err, "Error: Failed to execute kernel[0]");
    
	for(i = 1; i < N ; i ++){        
            clFinish(queue[i-1]);
            err = clEnqueueNDRangeKernel(queue[i], kernel[i], 3, NULL, global, NULL, 0, NULL, NULL);
            checkError(err, "Error: Failed to execute kernel[%d]",i);
            //err = clEnqueueReadBuffer(queue[i-1], d_act1[i-1], CL_TRUE, 0, sizeof(int) * 128 * 32 * 32, &h_act1[i-1], 0, NULL, NULL);
            //checkError(err, "Error: Failed to read kernel arguments! - kernel[%d] - d_act1",i-1);
            }
    clFinish(queue[N-1]);

    err = clEnqueueReadBuffer(queue[i-1], d_act1, CL_TRUE, 0, sizeof(int) * 128 * 32 * 32, h_act1, 0, NULL, NULL);
    checkError(err, "Error: Failed to read kernel arguments! - kernel[%d] - d_act1",N-1);
    err = clEnqueueReadBuffer(queue[i-1], d_fmap1, CL_TRUE, 0, sizeof(int) * 128 * 34 * 34, h_fmap1, 0, NULL, NULL);
    checkError(err, "Error: Failed to read kernel arguments! - kernel[%d] - d_act1",N-1);
    printf("Completed Execution \n");

    printf("Complete \n");

    correct=0;
    int count=0;
    int flag=0;

    for(unsigned char i = 0; i < 128; i++){
        for(unsigned char j = 0; j < 34; j++){
            for(unsigned char k = 0; k < 34; k++){
                count++;
                //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),w1[i][2][2][2], h_w1[ 2 + (2 * 3) + (2 * 3 * 3) + (i * 3 * 3 * 3)]);

                if(fmap1[i][j][k] == h_fmap1[ k + (j * 34) + (i * (34*34))]){
                    //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),act1[i][j][k], h_act1[ k + (j * 32) + (i * (32*32))]);
                    correct++;
                }
                else{
                    //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),act1[i][j][k], h_act1[ k + (j * 32) + (i * (32*32))]);
                    flag++;
                }

              //  if(flag > 0)
              //      printf("Matrix %d %d %d - %d\n",i,j,k,flag);
            }
        }
        if(flag > 0)
            printf("Matrix %d  - %d\n",i,flag);
        flag=0;
    }

    printf("\nNo. of Data Correct for act1  %d / %d\n",correct,count);

void cleanup();
}
void cleanup(){

for(i = 0; i < N ; i ++){        
    clReleaseMemObject(d_fmap0[i]);
    clReleaseMemObject(d_w1[i]);
    clReleaseMemObject(d_norm1[i]);
    clReleaseKernel(kernel[i]);
    clReleaseCommandQueue(queue[i]);
}
    clReleaseMemObject(d_act1);
    clReleaseProgram(program);
    clReleaseContext(context);

}
