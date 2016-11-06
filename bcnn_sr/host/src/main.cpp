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
#include "data1.h"
#include "results.h"
 
 using namespace aocl_utils;
#define N 9

unsigned int correct;
cl_int err;
cl_uint numPlatforms;

//int *X1 = (int*) memalign ( AOCL_ALIGNMENT, (sizeof(int)));

size_t global[3];                       // global domain size for our calculation
size_t local[3];                        // local domain size for our calculation

cl_platform_id platform;                // compute platform id
cl_device_id device;                    // compute device id
cl_context context;                     // compute context

cl_command_queue queue[N];             // compute command queue
cl_program program;                     // compute program
cl_kernel kernel[N];                    // compute kernel


char h_dim1[3];
short h_dim3[3];
short h_dim5[3];

int i,j;

scoped_array<cl_mem> d_fmap0;
scoped_array<cl_mem> d_norm1;
scoped_array<cl_mem> d_w1;
scoped_array<cl_mem> d_dim1;
cl_mem d_fmap1,d_act1;

scoped_array<cl_mem> d_norm2;
scoped_array<cl_mem> d_w2;
cl_mem d_fmap2,d_act2;

scoped_array<cl_mem> d_norm3;
scoped_array<cl_mem> d_w3;
scoped_array<cl_mem> d_dim3;
cl_mem d_fmap3,d_act3;

scoped_array<cl_mem> d_norm4;
scoped_array<cl_mem> d_w4;
cl_mem d_fmap4,d_act4;

scoped_array<cl_mem> d_norm5;
scoped_array<cl_mem> d_w5;
scoped_array<cl_mem> d_dim5;
cl_mem d_fmap5,d_act5;

scoped_array<cl_mem> d_norm6;
scoped_array<cl_mem> d_w6;
cl_mem d_fmap6,d_act6;

scoped_array<cl_mem> d_norm7;
scoped_array<cl_mem> d_w7;
cl_mem d_fmap7,d_act7;

scoped_array<cl_mem> d_norm8;
scoped_array<cl_mem> d_w8;
cl_mem d_fmap8,d_act8;

scoped_array<cl_mem> d_norm9;
scoped_array<cl_mem> d_w9;
cl_mem d_fmap9,d_act9;


scoped_aligned_ptr<char> h_fmap0;
scoped_aligned_ptr<char> h_w1;
scoped_aligned_ptr<short> h_norm1;
scoped_aligned_ptr<bool> h_fmap1;
scoped_aligned_ptr<int> h_act1;

scoped_aligned_ptr<bool> h_w2;
scoped_aligned_ptr<short> h_norm2;
scoped_aligned_ptr<short> h_act2;
scoped_aligned_ptr<bool> h_fmap2;

scoped_aligned_ptr<bool> h_w3;
scoped_aligned_ptr<short> h_norm3;
scoped_aligned_ptr<short> h_act3;
scoped_aligned_ptr<bool> h_fmap3;

scoped_aligned_ptr<bool> h_w4;
scoped_aligned_ptr<short> h_norm4;
scoped_aligned_ptr<short> h_act4;
scoped_aligned_ptr<bool> h_fmap4;

scoped_aligned_ptr<bool> h_w5;
scoped_aligned_ptr<short> h_norm5;
scoped_aligned_ptr<short> h_act5;
scoped_aligned_ptr<bool> h_fmap5;

scoped_aligned_ptr<bool> h_w6;
scoped_aligned_ptr<short> h_norm6;
scoped_aligned_ptr<short> h_act6;
scoped_aligned_ptr<bool> h_fmap6;

scoped_aligned_ptr<bool> h_w7;
scoped_aligned_ptr<short> h_norm7;
scoped_aligned_ptr<short> h_act7;
scoped_aligned_ptr<bool> h_fmap7;

scoped_aligned_ptr<bool> h_w8;
scoped_aligned_ptr<short> h_norm8;
scoped_aligned_ptr<short> h_act8;
scoped_aligned_ptr<bool> h_fmap8;

scoped_aligned_ptr<bool> h_w9;
scoped_aligned_ptr<short> h_norm9;
scoped_aligned_ptr<short> h_act9;
scoped_aligned_ptr<short> h_fmap9;


void cleanup();


int main(void){

//Layer 1
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
    d_fmap0.reset(1);
    d_norm1.reset(1);
    d_w1.reset(1);
    d_dim1.reset(1);
    h_act1.reset(128*34*34);
//Layer 2
    h_w2.reset(128*128*3*3);
    for(i = 0 ; i < 128*128*3*3 ; i ++){
        h_w2[i] = w2[i];
    }
    h_norm2.reset(128);
    for(i = 0 ; i < 128 ; i ++){
        h_norm2[i] = norm2[i];
    }
    h_fmap2.reset(128*18*18);
    h_act2.reset(128*32*32);
    d_norm2.reset(1);
    d_w2.reset(1);
//Layer 3
    h_w3.reset(256*128*3*3);
    for(i = 0 ; i < 256*128*3*3 ; i ++){
        h_w3[i] = w3[i];
    }
    h_norm3.reset(256);
    for(i = 0 ; i < 256 ; i ++){
        h_norm3[i] = norm3[i];
    }
    h_fmap3.reset(256*18*18);
    h_act3.reset(256*16*16);
    d_norm3.reset(1);
    d_w3.reset(1);
    d_dim3.reset(1);

//Layer 4
    h_w4.reset(256*256*3*3);
    for(i = 0 ; i < 256*256*3*3 ; i++){
        h_w4[i] = w4[i];
    }
    h_norm4.reset(256);
    for(i = 0 ; i < 256 ; i ++){
        h_norm4[i] = norm4[i];
    }
    h_fmap4.reset(256*10*10);
    h_act4.reset(256*16*16);
    d_norm4.reset(1);
    d_w4.reset(1);

//Layer 5
    h_w5.reset(512*256*3*3);
    for(i = 0 ; i < 512*256*3*3 ; i++){
        h_w5[i] = w5[i];
    }
    h_norm5.reset(512);
    for(i = 0 ; i < 512 ; i ++){
        h_norm5[i] = norm5[i];
    }
    h_fmap5.reset(512*10*10);
    h_act5.reset(512*8*8);
    d_norm5.reset(1);
    d_w5.reset(1);
    d_dim5.reset(1); 

//Layer 6
    h_w6.reset(512*512*3*3);
    for(i = 0 ; i < 512*512*3*3 ; i++){
        h_w6[i] = w6[i];
    }
    h_norm6.reset(512);
    for(i = 0 ; i < 512 ; i ++){
        h_norm6[i] = norm6[i];
    }
    h_fmap6.reset(8192);
    h_act6.reset(512*8*8);
    d_norm6.reset(1);
    d_w6.reset(1);

//Layer 7
    h_w7.reset(8192*1024);
    for(i = 0 ; i < 8192*1024 ; i++){
        h_w7[i] = w7[i];
    }
    h_norm7.reset(1024);
    for(i = 0 ; i < 1024 ; i ++){
        h_norm7[i] = norm7[i];
    }
    h_fmap7.reset(1024);
    h_act7.reset(1024);
    d_norm7.reset(1);
    d_w7.reset(1);

//Layer 8
    h_w8.reset(1024*1024);
    for(i = 0 ; i < 1024*1024 ; i++){
        h_w8[i] = w8[i];
    }
    h_norm8.reset(1024);
    for(i = 0 ; i < 1024 ; i ++){
        h_norm8[i] = norm8[i];
    }
    h_fmap8.reset(1024);
    h_act8.reset(1024);
    d_norm8.reset(1);
    d_w8.reset(1);

//Layer 9
    h_w9.reset(1024*10);
    for(i = 0 ; i < 1024*10 ; i++){
        h_w9[i] = w9[i];
    }
    h_norm9.reset(10);
    for(i = 0 ; i < 10 ; i ++){
        h_norm9[i] = norm9[i];
    }
    h_fmap9.reset(10);
    h_act9.reset(10);
    d_norm9.reset(1);
    d_w9.reset(1);     
             

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

    string binary_file = getBoardBinaryFile("kernel_e2", device);
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


    d_fmap0[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * 3 * 34 * 34, NULL, NULL);
    d_w1[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char) * 128 * 3 * 3 * 3, NULL, NULL);
    d_norm1[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 128, NULL, NULL);
    d_dim1[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char)*3, NULL, NULL);
    
    d_act1 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * 128 * 34 * 34, NULL, NULL);

    d_fmap1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 128 * 34 * 34, NULL, NULL);
    
    h_dim1 = {128,33,33};

    d_w2[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 128 * 128 * 3 * 3, NULL, NULL);
    d_norm2[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 128, NULL, NULL);
    d_act2 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 128 * 32 * 32, NULL, NULL);
    d_fmap2 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 128 * 18 * 18, NULL, NULL);

    d_w3[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 256 * 128 * 3 * 3, NULL, NULL);
    d_norm3[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 256, NULL, NULL);
    d_dim3[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short)*3, NULL, NULL);
    d_act3 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 256 * 16 * 16, NULL, NULL);
    d_fmap3 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 256 * 18 * 18, NULL, NULL);
    
    h_dim3 = {256,17,17};

    d_w4[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 256 * 256 * 3 * 3, NULL, NULL);
    d_norm4[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 256, NULL, NULL);
    d_act4 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 256 * 16 * 16, NULL, NULL);
    d_fmap4 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 256 * 10 * 10, NULL, NULL);

    d_w5[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 512 * 256 * 3 * 3, NULL, NULL);
    d_norm5[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 512, NULL, NULL);
    d_dim5[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 3, NULL, NULL);
    d_act5 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 512 * 8 * 8, NULL, NULL);
    d_fmap5 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 512 * 10 * 10, NULL, NULL);
    
    h_dim5 = {512,9,9}; 

    d_w6[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 512 * 512 * 3 * 3, NULL, NULL);
    d_norm6[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 512, NULL, NULL);
    d_act6 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 512 * 8 * 8, NULL, NULL);
    d_fmap6 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 8192, NULL, NULL);

    d_w7[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 8192 * 1024, NULL, NULL);
    d_norm7[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 1024, NULL, NULL);
    d_act7 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 1024, NULL, NULL);
    d_fmap7 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 1024, NULL, NULL);

    d_w8[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 1024 * 1024, NULL, NULL);
    d_norm8[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 1024, NULL, NULL);
    d_act8 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 1024, NULL, NULL);
    d_fmap8 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(bool) * 1024, NULL, NULL); 

    d_w9[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(bool) * 10 * 1024, NULL, NULL);
    d_norm9[0] = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(short) * 10, NULL, NULL);
    d_act9 = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(short) * 10, NULL, NULL);
    d_fmap9 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(short) * 10, NULL, NULL);    
    // printf("Completed Buffer Creation \n");
    cl_event event_kernel[N];

    


        queue[0] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Error: Failed to create a command queue[%d]!",0);  
        
        // Create the compute kernel in the program we wish to run
        //
        kernel[0] = clCreateKernel(program, "read_data", &err);
        checkError(err, "Error: Failed to create compute kernel[%d]!",0);

        queue[1] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Error: Failed to create a command queue[%d]!",1);  

        kernel[1] = clCreateKernel(program, "layer_one", &err);
        checkError(err, "Error: Failed to create compute kernel[%d]!",1);

        queue[2] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Error: Failed to create a command queue[%d]!",3);  

        kernel[2] = clCreateKernel(program, "add_border", &err);
        checkError(err, "Error: Failed to create compute kernel[%d]!",3);

        queue[3] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        checkError(err, "Error: Failed to create a command queue[%d]!",2);  

        kernel[3] = clCreateKernel(program, "write_data", &err);
        checkError(err, "Error: Failed to create compute kernel[%d]!",2);

        // queue[4] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // checkError(err, "Error: Failed to create a command queue[%d]!",4);  

        // kernel[4] = clCreateKernel(program, "layerfive", &err);
        // checkError(err, "Error: Failed to create compute kernel[%d]!",4);

        // queue[5] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // checkError(err, "Error: Failed to create a command queue[%d]!",5);  

        // kernel[5] = clCreateKernel(program, "layersix", &err);
        // checkError(err, "Error: Failed to create compute kernel[%d]!",5);

        // queue[6] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // checkError(err, "Error: Failed to create a command queue[%d]!",6);  

        // kernel[6] = clCreateKernel(program, "layerseven", &err);
        // checkError(err, "Error: Failed to create compute kernel[%d]!",6); 

        // queue[7] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // checkError(err, "Error: Failed to create a command queue[%d]!",7);  

        // kernel[7] = clCreateKernel(program, "layereight", &err);
        // checkError(err, "Error: Failed to create compute kernel[%d]!",7); 

        // queue[8] = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
        // checkError(err, "Error: Failed to create a command queue[%d]!",8);  

        // kernel[8] = clCreateKernel(program, "layernine", &err);
        // checkError(err, "Error: Failed to create compute kernel[%d]!",8); 

//Layer 1
        // Write our data set into the input array in device memory
        //
        err = clEnqueueWriteBuffer(queue[0], d_fmap0[0], CL_FALSE, 0, sizeof(char) * 3 * 34 * 34, h_fmap0, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_fmap0",0);

        err = clEnqueueWriteBuffer(queue[1], d_w1[0], CL_FALSE, 0, sizeof(char) * 128 * 3 * 3 * 3, h_w1, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w1",0);

        err = clEnqueueWriteBuffer(queue[1], d_norm1[0], CL_FALSE, 0, sizeof(short) * 128, h_norm1, 0, NULL, NULL);
        checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm1",0);

        // err = clEnqueueWriteBuffer(queue[0], d_dim1[0], CL_FALSE, 0, sizeof(char) * 3, h_dim1, 0, NULL, NULL);
        // checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_debug",0);    
      
// //Layer 2
//         err = clEnqueueWriteBuffer(queue[1], d_w2[0], CL_FALSE, 0, sizeof(bool) * 128 * 128 * 3 * 3, h_w2, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w2",1);

//         err = clEnqueueWriteBuffer(queue[1], d_norm2[0], CL_FALSE, 0, sizeof(short) * 128, h_norm2, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm2",1);

// //Layer 3        
//         err = clEnqueueWriteBuffer(queue[2], d_w3[0], CL_FALSE, 0, sizeof(bool) * 256 * 128 * 3 * 3, h_w3, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w3",2);

//         err = clEnqueueWriteBuffer(queue[2], d_norm3[0], CL_FALSE, 0, sizeof(short) * 256, h_norm3, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm3",2);

//         err = clEnqueueWriteBuffer(queue[2], d_dim3[0], CL_FALSE, 0, sizeof(short) * 3, h_dim3, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_dim3",2);

// //Layer 4
//         err = clEnqueueWriteBuffer(queue[3], d_w4[0], CL_FALSE, 0, sizeof(bool) * 256 * 256 * 3 * 3, h_w4, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w4",3);

//         err = clEnqueueWriteBuffer(queue[3], d_norm4[0], CL_FALSE, 0, sizeof(short) * 256, h_norm4, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm4",3);

// //Layer 5        
//         err = clEnqueueWriteBuffer(queue[4], d_w5[0], CL_FALSE, 0, sizeof(bool) * 512 * 256 * 3 * 3, h_w5, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w5",4);

//         err = clEnqueueWriteBuffer(queue[4], d_norm5[0], CL_FALSE, 0, sizeof(short) * 512, h_norm5, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm5",4);

//         err = clEnqueueWriteBuffer(queue[4], d_dim5[0], CL_FALSE, 0, sizeof(short) * 3, h_dim5, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_dim5",4);

// //Layer 6
//         err = clEnqueueWriteBuffer(queue[5], d_w6[0], CL_FALSE, 0, sizeof(bool) * 512 * 512 * 3 * 3, h_w6, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w6",5);

//         err = clEnqueueWriteBuffer(queue[5], d_norm6[0], CL_FALSE, 0, sizeof(short) * 512, h_norm6, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm6",5);

// //Layer 7
//         err = clEnqueueWriteBuffer(queue[6], d_w7[0], CL_FALSE, 0, sizeof(bool) * 8192 *1024, h_w7, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w7",6);

//         err = clEnqueueWriteBuffer(queue[6], d_norm7[0], CL_FALSE, 0, sizeof(short) * 1024, h_norm7, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm7",6);   

// //Layer 8
//         err = clEnqueueWriteBuffer(queue[7], d_w8[0], CL_FALSE, 0, sizeof(bool) * 1024 *1024, h_w8, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w8",7);

//         err = clEnqueueWriteBuffer(queue[7], d_norm8[0], CL_FALSE, 0, sizeof(short) * 1024, h_norm8, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm8",7); 

// //Layer 9
//         err = clEnqueueWriteBuffer(queue[8], d_w9[0], CL_FALSE, 0, sizeof(bool) * 10 *1024, h_w9, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_w9",8);

//         err = clEnqueueWriteBuffer(queue[8], d_norm9[0], CL_FALSE, 0, sizeof(short) * 10, h_norm9, 0, NULL, NULL);
//         checkError(err, "Error: Failed to copy kernel arguments! - kernel[%d] - h_norm9",8); 
//Layer 1
        // Set the arguments to our compute kernel
        //
        unsigned argi = 0;
        err = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &d_fmap0[0]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap0",0);
        
        argi = 0;
        err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_fmap0[0]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap0",1);
        
        err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_w1[0]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w1",1);

        err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_norm1[0]);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm1",1);

        // err = clSetKernelArg(kernel[0], argi++, sizeof(cl_mem), &d_dim1[0]);
        // checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_debug",0);

        err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_fmap1);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap1",1);

        err = clSetKernelArg(kernel[3], 0, sizeof(cl_mem), &d_act1);
        checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act1",3);
//Layer 2
//         argi = 0;
//         err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_fmap1);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap1",1);

//         err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_w2[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w2",1);

//         err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_norm2[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm2",1);

//         err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_act2);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act2",1);

//         err = clSetKernelArg(kernel[1], argi++, sizeof(cl_mem), &d_fmap2);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap2",1);
// //Layer 3
//         argi = 0;
//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_fmap2);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap2",2);

//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_w3[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w3",2);

//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_norm3[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm3",2);

//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_dim3[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_dim3",2);

//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_act3);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act3",2);

//         err = clSetKernelArg(kernel[2], argi++, sizeof(cl_mem), &d_fmap3);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap3",2);

// //Layer 4
//         argi = 0;
//         err = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &d_fmap3);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap3",3);

//         err = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &d_w4[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w4",3);

//         err = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &d_norm4[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm4",3);

//         err = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &d_act4);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act4",3);

//         err = clSetKernelArg(kernel[3], argi++, sizeof(cl_mem), &d_fmap4);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap4",3);

// //Layer 5
//         argi = 0;
//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_fmap4);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap4",4);

//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_w5[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w5",4);

//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_norm5[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm5",4);

//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_dim5[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_dim5",4);

//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_act5);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act5",4);

//         err = clSetKernelArg(kernel[4], argi++, sizeof(cl_mem), &d_fmap5);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap5",4);

// //Layer 6
//         argi = 0;
//         err = clSetKernelArg(kernel[5], argi++, sizeof(cl_mem), &d_fmap5);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap5",5);

//         err = clSetKernelArg(kernel[5], argi++, sizeof(cl_mem), &d_w6[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w6",5);

//         err = clSetKernelArg(kernel[5], argi++, sizeof(cl_mem), &d_norm6[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm6",5);

//         err = clSetKernelArg(kernel[5], argi++, sizeof(cl_mem), &d_act6);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act6",5);

//         err = clSetKernelArg(kernel[5], argi++, sizeof(cl_mem), &d_fmap6);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap6",5);

// //Layer 7
//         argi = 0;
//         err = clSetKernelArg(kernel[6], argi++, sizeof(cl_mem), &d_fmap6);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap6",6);

//         err = clSetKernelArg(kernel[6], argi++, sizeof(cl_mem), &d_w7[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w7",6);

//         err = clSetKernelArg(kernel[6], argi++, sizeof(cl_mem), &d_norm7[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm7",6);

//         err = clSetKernelArg(kernel[6], argi++, sizeof(cl_mem), &d_act7);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act7",6);

//         err = clSetKernelArg(kernel[6], argi++, sizeof(cl_mem), &d_fmap7);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap7",6);

// //Layer 8
//         argi = 0;
//         err = clSetKernelArg(kernel[7], argi++, sizeof(cl_mem), &d_fmap7);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap7",7);

//         err = clSetKernelArg(kernel[7], argi++, sizeof(cl_mem), &d_w8[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w8",7);

//         err = clSetKernelArg(kernel[7], argi++, sizeof(cl_mem), &d_norm8[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm8",7);

//         err = clSetKernelArg(kernel[7], argi++, sizeof(cl_mem), &d_act8);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act8",7);

//         err = clSetKernelArg(kernel[7], argi++, sizeof(cl_mem), &d_fmap8);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap8",7);

// //Layer 9
//         argi = 0;
//         err = clSetKernelArg(kernel[8], argi++, sizeof(cl_mem), &d_fmap8);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap8",8);

//         err = clSetKernelArg(kernel[8], argi++, sizeof(cl_mem), &d_w9[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_w9",8);

//         err = clSetKernelArg(kernel[8], argi++, sizeof(cl_mem), &d_norm9[0]);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_norm9",8);

//         err = clSetKernelArg(kernel[8], argi++, sizeof(cl_mem), &d_act9);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_act9",8);

//         err = clSetKernelArg(kernel[8], argi++, sizeof(cl_mem), &d_fmap9);
//         checkError(err, "Error: Failed to set kernel arguments! - kernel[%d] - d_fmap9",8);                                                
    
     printf("Completed Setting Arguments \n");

    // global = {34, 34, 128};    
    // err = clEnqueueNDRangeKernel(queue[0], kernel[0], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[0]");
    
    // clFinish(queue[0]);
    
    // global = {32, 32, 128};
    // err = clEnqueueNDRangeKernel(queue[1], kernel[1], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[1]");

    // clFinish(queue[1]);

    // global = {18, 18, 256};
    // err = clEnqueueNDRangeKernel(queue[2], kernel[2], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[2]");

    // clFinish(queue[2]);

    // global = {16, 16, 256};
    // err = clEnqueueNDRangeKernel(queue[3], kernel[3], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[3]");

    // clFinish(queue[3]);

    // global = {10, 10, 512};
    // err = clEnqueueNDRangeKernel(queue[4], kernel[4], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[4]");

    // clFinish(queue[4]);

    // global = {8, 8, 512};
    // err = clEnqueueNDRangeKernel(queue[5], kernel[5], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[5]");

    // clFinish(queue[5]);

    // global = {1024, 1, 1};
    // err = clEnqueueNDRangeKernel(queue[6], kernel[6], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[6]");

    // clFinish(queue[6]);

    // global = {1024, 1, 1};
    // err = clEnqueueNDRangeKernel(queue[7], kernel[7], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[7]");

    // clFinish(queue[7]);

    // global = {10, 1, 1};
    // err = clEnqueueNDRangeKernel(queue[8], kernel[8], 3, NULL, global, NULL, 0, NULL, NULL);
    // checkError(err, "Error: Failed to execute kernel[6]");

    // clFinish(queue[8]);

    // printf("Completed Execution \n");
    
    // err = clEnqueueReadBuffer(queue[8], d_fmap9, CL_TRUE, 0, sizeof(short) * 10, h_fmap9, 0, NULL, NULL);
    // checkError(err, "Error: Failed to read kernel arguments! - kernel[%d] - d_fmap9",8);
    err = clEnqueueTask(queue[0], kernel[0], 0, NULL, NULL);
    checkError(err, "Error: Failed to execute kernel[0]");

    err = clEnqueueTask(queue[1], kernel[1], 0, NULL, NULL);
    checkError(err, "Error: Failed to execute kernel[1]");

    err = clEnqueueTask(queue[2], kernel[2], 0, NULL, NULL);
    checkError(err, "Error: Failed to execute kernel[2]");

    err = clEnqueueTask(queue[3], kernel[3], 0, NULL, NULL);
    checkError(err, "Error: Failed to execute kernel[3]");
    
    clFinish(queue[3]);
    
    err = clEnqueueReadBuffer(queue[3], d_act1, CL_TRUE, 0, sizeof(int) * 34*34*128, h_act1, 0, NULL, NULL);
    checkError(err, "Error: Failed to read kernel arguments! - kernel[%d] - h_act1",3);

    printf("Complete \n");

    correct=0;
    int count=0;
    int flag=0;

     for(int i = 0; i < 1; i++){
         for(int j = 1; j < 2; j++){
             for(int k = 0; k < 34; k++){
                 printf("%d\n",h_act1[k + (j * 34) + (i * (34*34))]);
                 //count++;
                 //if(h_fmap0[i][j][k] == h_fmap0_1[ k + (j * 34) + (i * (34*34))]){
                    //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 8) + (i * (8*8))),act5[i][j][k], h_act5[ k + (j * 8) + (i * (8*8))]);
                   // correct++;
               // }

             }
         }
     }

                 //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),w1[i][2][2][2], h_w1[ 2 + (2 * 3) + (2 * 3 * 3) + (i * 3 * 3 * 3)]);

    // for(int i = 0; i < 512; i++){
    //     for(int j = 0; j < 8; j++){
    //         for(int k = 0; k < 8; k++){
    //             count++;
    //             //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),w1[i][2][2][2], h_w1[ 2 + (2 * 3) + (2 * 3 * 3) + (i * 3 * 3 * 3)]);

    //             if(act6[i][j][k] == h_act6[ k + (j * 8) + (i * (8*8))]){
    //                 //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 8) + (i * (8*8))),act5[i][j][k], h_act5[ k + (j * 8) + (i * (8*8))]);
    //                 correct++;
    //             }
    //             else{
    //                 printf("Index %d ->> Expected = %d  Optained = %d\n", (k + (j * 8) + (i * (8*8))),act6[i][j][k], h_act6[k + (j * 8) + (i * (8*8))]);
    //                 //if((j > 0 && j < 17) && (k > 0 && k < 17))
    //                 //printf("Index %d ->> Expected = %d  Optained = % d\n",((k-1) + ((j-1) * 16) + (i * (16*16))),act3[i][j-1][k-1], h_act3[((k-1) + ((j-1) * 16) + (i * (16*16)))]);

    //                 flag++;
    //             }

    //                // if(flag > 0)
    //                //     printf("Matrix %d %d %d - %d\n",i,j,k,flag);
    //         }
    //     }
        
    //      if(flag > 0)
    //          printf("Matrix %d  - %d\n",i,flag);
    //     flag=0;
    // }

    //    for(int i = 0; i < 10; i++){
    //             count++;
    //             //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 32) + (i * (32*32))),w1[i][2][2][2], h_w1[ 2 + (2 * 3) + (2 * 3 * 3) + (i * 3 * 3 * 3)]);

    //             if(fmap9[i] == h_fmap9[i]){
    //                 //printf("Index %d ->> Expected = %d  Optained = %d\n",(k + (j * 8) + (i * (8*8))),act5[i][j][k], h_act5[ k + (j * 8) + (i * (8*8))]);
    //                 correct++;
    //             }
    //             else{
    //                 printf("Index %d ->> Expected = %d  Optained = %d\n",i,fmap9[i], h_fmap9[i]);
    //                 //if((j > 0 && j < 17) && (k > 0 && k < 17))
    //                 //printf("Index %d ->> Expected = %d  Optained = %d\n",((k-1) + ((j-1) * 16) + (i * (16*16))),act3[i][j-1][k-1], h_act3[((k-1) + ((j-1) * 16) + (i * (16*16)))]);
    //             }

    // }
    //  printf("No. of Data Correct for fmap9  %d / %d\n",correct,count);

void cleanup();
}


void cleanup(){

    for(i = 0; i < N ; i ++){        

        clReleaseKernel(kernel[i]);
        clReleaseCommandQueue(queue[i]);
    }

    clReleaseMemObject(d_fmap0[0]);
    clReleaseMemObject(d_w1[0]);
    clReleaseMemObject(d_norm1[0]);

    clReleaseMemObject(d_fmap1);
    clReleaseMemObject(d_w2[0]);
    clReleaseMemObject(d_norm2[0]);
    clReleaseMemObject(d_fmap2);
    
    clReleaseMemObject(d_w3[0]);
    clReleaseMemObject(d_norm3[0]);
    clReleaseMemObject(d_fmap3);

    clReleaseMemObject(d_w4[0]);
    clReleaseMemObject(d_norm4[0]);
    clReleaseMemObject(d_fmap4);

    clReleaseMemObject(d_w5[0]);
    clReleaseMemObject(d_norm5[0]);
    clReleaseMemObject(d_fmap5);

    clReleaseMemObject(d_w6[0]);
    clReleaseMemObject(d_norm6[0]);
    clReleaseMemObject(d_fmap6);

    clReleaseMemObject(d_w7[0]);
    clReleaseMemObject(d_norm7[0]);
    clReleaseMemObject(d_fmap7);
    
    // free(h_fmap1);
    // free(h_fmap2);
    // free(h_fmap3);
    // free(h_fmap4);
    // free(h_fmap5);
    // free(h_fmap6);
    // free(h_fmap7);

    // free(h_act2);
    // free(h_act3);
    // free(h_act4);
    // free(h_act5);
    // free(h_act6);
    // free(h_act7);
    
    clReleaseProgram(program);
    clReleaseContext(context);

}
