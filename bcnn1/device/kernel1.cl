
__kernel void conv( __global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_debug,
__global int* d_offset,
__global int *restrict d_act1,
__global int *restrict d_fmap1
){

	int fnum1, hei1, wid1;
	int i1, j1, k1, temp, fmap, w, index, index1;
    local int act1;
     	   fnum1 = get_global_id(2)+ *d_offset;
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);

           index1 = (wid1) + ((hei1) * 34) + (fnum1 * 34 * 34);

if(fnum1 < d_debug[0] && (hei1 > 0 && hei1 < d_debug[1]) && (wid1 > 0 && wid1 < d_debug[2])){
            index = (wid1-1) + ((hei1-1) * 32) + (fnum1 * 32 * 32);
            act1 = 0;
            for(i1 = 0; i1 < 3; i1++){
                for(j1 = 0; j1 < 3; j1++){
                    for(k1 = 0; k1 < 3; k1++){

                            fmap = ((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)));
                            w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));

                            temp = d_fmap0[fmap] * d_w1[w];

                            act1 = act1 + temp;

                            printf("index %d index1 %d fmap %d w %d temp %d act1 %d\n",index,index1,fmap,w,temp,act1);

                        }
                    }
                }
                d_act1[index]=act1;
                // Normalization and non-linearity
                if(d_act1[index] > d_norm1[fnum1])
                    d_fmap1[index1] = 1;
                else
                    d_fmap1[index1] = 0;

    }
    else
        d_fmap1[index1] = 0;
}

