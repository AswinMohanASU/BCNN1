__kernel void reduce_sum(__global int* input, 
	__global int* accumulator) {
  
  		   int i = get_global_id(0);
           int j = get_global_id(1);
           int k = get_global_id(2);
  		
  		accumulator[k + (j * 34) + (i * (34*34))] = input[k + (j * 34) + (i * (34*34))];
  		printf("%d\n",accumulator[k + (j * 34) + (i * (34*34))]);
  
}