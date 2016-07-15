__kernel void reduce_sum(__global int* input, 
	__global int* accumulator) {
  
  for(int i = 0 ; i < 3468 ; i++){
  		accumulator[i] = input[i];
  		printf("%d \n",accumulator[i]);
  }
}