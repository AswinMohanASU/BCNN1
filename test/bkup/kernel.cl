__kernel void test (__global int *fmap0,
__global int *restrict x
){

	int fnum1, hei1, wid1;

     	   fnum1 = get_global_id(0);
           hei1 = get_global_id(1);
           wid1 = get_global_id(2);

	x[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))] = fmap0[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))];

}