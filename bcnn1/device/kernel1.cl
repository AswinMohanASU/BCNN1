
__kernel void conv( __global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_debug,
__global int* d_offset,
__global int *restrict d_act1,
__global int *restrict d_fmap1
){

	int fnum1, hei1, wid1;
	int i1, j1, k1, temp, fmap, w, act_index, act_index1,index;

     	   fnum1 = get_global_id(2)+ *d_offset;
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);
           index = wid1 + (hei1 * 34) + (fnum1 * 34 * 34);
           d_fmap1[index] = 0;

if(fnum1 < d_debug[0] && hei1 < d_debug[1] && wid1 < d_debug[2]){
           act_index = wid1 + (hei1 * 32) + (fnum1 * 32 * 32);
           act_index1 = (wid1+1) + ((hei1+1) * 34) + (fnum1 * 34 * 34);

            d_act1[act_index] = 0;
            for(i1 = 0; i1 < 3; i1++){
                for(j1 = 0; j1 < 3; j1++){
                    for(k1 = 0; k1 < 3; k1++){

                            fmap = ((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)));
                            w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));

                            temp = d_fmap0[fmap] * d_w1[w];

                            d_act1[act_index] = d_act1[act_index] + temp;

                        }
                    }
                }

                // Normalization and non-linearity
                if(d_act1[act_index] > d_norm1[fnum1])
                    d_fmap1[act_index1] = 1;
                else
                    d_fmap1[act_index1] = 0;

    }
}

