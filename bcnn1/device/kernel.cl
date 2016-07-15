__global fmap1[128 * 34 * 34];

__kernel void conv(__global int* fmap0,
__global int* w1,
__global int* norm1
__global int *restrict act1){

	int fnum1, hei1, wid1;
	int i1, j1, k1, temp;

     	   fnum1 = get_global_id(0);
           hei1 = get_global_id(1);
           wid1 = get_global_id(2);


 for(i1 = 0; i1 < 3; i1++){
 	for(j1 = 0; j1 < 3; j1++){
 		for(k1 = 0; k1 < 3; k1++){

 				temp = fmap0[((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)))] * w1[k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3)];
 				act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] = act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] + temp;

            }
        }
    }

    // Normalization and non-linearity
    if(act1[fnum1][hei1][wid1] > norm1[fnum1])
        fmap1[fnum1][hei1+1][wid1+1] = 1;
    else
        fmap1[fnum1][hei1+1][wid1+1] = 0;

}

