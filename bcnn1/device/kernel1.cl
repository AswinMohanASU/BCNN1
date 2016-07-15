__global int fmap1[128][34][34];

__kernel void conv( __global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_act1
){

	int fnum1, hei1, wid1;
	int i1, j1, k1, temp, fmap, w, index;

     	   fnum1 = get_global_id(0);
           hei1 = get_global_id(1);
           wid1 = get_global_id(2);
           index = wid1 + (hei1 * 32) + (fnum1 * 32 * 32);

           d_act1[index] = 0;

 for(i1 = 0; i1 < 3; i1++){
 	for(j1 = 0; j1 < 3; j1++){
 		for(k1 = 0; k1 < 3; k1++){

                fmap = ((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)));
                w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));

 				temp = d_fmap0[fmap] * d_w1[w];

                d_act1[index] = d_act1[index] + temp;

                printf("Index [%d][%d][%d] --> fmap=%d w=%d  temp=%d  d_act1[%d]= %d",i1,j1,k1,fmap,w,index,d_act1[index]);

            }
        }
    }

    // Normalization and non-linearity
    if(d_act1[index] > d_norm1[fnum1])
        fmap1[fnum1][hei1+1][wid1+1] = 1;
    else
        fmap1[fnum1][hei1+1][wid1+1] = 0;

}

