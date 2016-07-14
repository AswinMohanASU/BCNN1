/**
* Utility Header File
*
* @author  Aswin Gunavelu Mohan <gmaswin@gmail.com>
* @version 1.0
* @since   2016-06-08
*/
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "CL/opencl.h"
#include "AOCL_Utils.h"

using namespace aocl_utils;
using namespace std;


//Function to return the Plaform ID using the Platform Name
cl_platform_id find_Platform(char* platformName){
	cl_int err;
	cl_uint num_of_platforms = 0;
	cl_uint selected_platform_index = num_of_platforms;
	err = clGetPlatformIDs(0, 0, &num_of_platforms);

	cout << "Number of available platforms: " << num_of_platforms << endl;
	cl_platform_id* platforms = new cl_platform_id[num_of_platforms];
	err = clGetPlatformIDs(num_of_platforms, platforms, 0);

	for (cl_uint i = 0; i < num_of_platforms; ++i)
	{
		// Get the length for the i-th platform name
		size_t platform_name_length = 0;
		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			0,
			0,
			&platform_name_length
			);


		// Get the name itself for the i-th platform
		char* platform_name = new char[platform_name_length];
		err = clGetPlatformInfo(
			platforms[i],
			CL_PLATFORM_NAME,
			platform_name_length,
			platform_name,
			0
			);

		cout << "    [" << i << "] " << platform_name;

		// decide if this i-th platform is what we are looking for
		// we select the first one matched skipping the next one if any
		if (
			strstr(platform_name, platformName) &&
			selected_platform_index == num_of_platforms // have not selected yet
			)
		{
			cout << " [Selected]";
			selected_platform_index = i;
			// do not stop here, just see all available platforms
		}

		cout << endl;
		delete[] platform_name;
	}

	if (selected_platform_index == num_of_platforms)
	{
		cerr
			<< "There is no found platform with name containing \""
			<< platformName << "\" as a substring.\n";

		return NULL;
	}

	cl_platform_id platform = platforms[selected_platform_index];

	return platform;
}

//Function to Compare Two Floats
bool Comparefloat2(float A, float B){
	int diff = (int )A*100000000 - (int )B*100000000;
	return (diff == 0);
}

//Function to Generate Random No.
float rand_float() {
  return float(rand()) / float(RAND_MAX) * 20.0f - 10.0f;
}


//Function to Check Error
void checkerror(cl_int err, char* errorMessage){
	if (err != CL_SUCCESS)
	{
		printf("%s\n",errorMessage);
		exit(err);
	}
}

