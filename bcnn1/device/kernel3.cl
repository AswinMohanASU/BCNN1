__global int fmap1[128][34][34];

__kernel void conv( __global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_act1
){

	int fnum1, hei1, wid1;
	int i1, j1, k1, temp;

     for(fnum1=0 ; fnum1 < 128 ; fnum++){
         for(hei1=0 ; hei1 < 32; hei1++){
            for(wid1=0 ; wid1 < 32; wid1++){

               d_act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] = 0;
                 for(i1 = 0; i1 < 3; i1++){
                    for(j1 = 0; j1 < 3; j1++){
                        for(k1 = 0; k1 < 3; k1++){

                                temp = d_fmap0[((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)))] * d_w1[k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3)];

                                d_act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] = d_act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] + temp;

                            }
                        }
                    }

                    // Normalization and non-linearity
                    if(d_act1[wid1 + (hei1 * 32) + (fnum1 * 32 * 32)] > d_norm1[fnum1])
                        fmap1[fnum1][hei1+1][wid1+1] = 1;
                    else
                        fmap1[fnum1][hei1+1][wid1+1] = 0;

            }
         }
     }


}

