int poolnorm (int a1, int a2, int a3, int a4, int norm){
	int s1,s2,s3;
	int dout;

	if (a1 >= a2)
		s1 = a1;
	else
		s1 = a2;

	if (a3 >= a4)
		s2 = a3;
	else
		s2 = a4;

	if (s1 >= s2)
		s3 = s1;
	else
		s3 = s2;

	if (norm >= s3)
		dout = 0;
	else
		dout = 1;

	return dout;
}

__kernel void conv( __global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_dim,
__global int *restrict d_fmap1
){

  int fnum1, hei1, wid1;
  int i1, j1, k1, temp, fmap, w, index;
    local int act1;
           fnum1 = get_global_id(2);
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);

           index = (wid1) + ((hei1) * 34) + (fnum1 * 34 * 34);

if(fnum1 < d_dim[0] && (hei1 > 0 && hei1 < d_dim[1]) && (wid1 > 0 && wid1 < d_dim[2])){
            
            act1 = 0;
            for(i1 = 0; i1 < 3; i1++){
                for(j1 = 0; j1 < 3; j1++){
                    for(k1 = 0; k1 < 3; k1++){

                            fmap = ((wid1+k1-1) + ((hei1+j1-1) * 34) + (i1 * (34 * 34)));
                            w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));

                            temp = d_fmap0[fmap] * d_w1[w];

                            act1 = act1 + temp;

                        }
                    }
                }
                // Normalization and non-linearity
                if(act1 > d_norm1[fnum1])
                    d_fmap1[index] = 1;
                else
                    d_fmap1[index] = 0;
    
    }
    else
        d_fmap1[index] = 0;
}