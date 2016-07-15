__kernel void reduce_sum(__global int* input, 
	__global int* accumulator) {
  
  		int i = get_global_id(0);
  		accumulator[i] = input[i];
  		
  
}