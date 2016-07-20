__global int fmap1[128 * 34 * 34];
__kernel void initialize(){
		       /*int fnum1 = get_global_id(2);
           int hei1 = get_global_id(0);
           int wid1 = get_global_id(1);
           int index = wid1 + (hei1 * 34) + (fnum1 * 34 * 34); */
           for(int i=0; i < 128 * 34 * 34 ; i++)
                fmap1[i]=0;
}
__kernel void conv(__global int *restrict d_fmap0,
__global int *restrict d_w1,
__global int *restrict d_norm1,
__global int *restrict d_debug,
__global int* d_offset,
__global int *restrict d_act1
){

	int fnum1, hei1, wid1;
	int i1, j1, k1, fmap, w, index;
  local int act1;
     	   fnum1 = get_global_id(2)+ *d_offset;
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);
        
index = wid1 + (hei1 * 32) + (fnum1 * 32 * 32);
if(fnum1 < d_debug[0] && hei1 < d_debug[1] && wid1 < d_debug[2]){

            act1 = 0;
            for(i1 = 0; i1 < 3; i1++){
                for(j1 = 0; j1 < 3; j1++){
                    for(k1 = 0; k1 < 3; k1++){

                            fmap = ((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)));
                            w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));

                            act1 = act1 + (d_fmap0[fmap] * d_w1[w]);

                        }
                    }
                }
                d_act1[index]=act1;

                // Normalization and non-linearity
                if(act1 > d_norm1[fnum1]){
               //     fmap1[fnum1][hei1+1][wid1+1] = 1;
                 }
                else{
               //     fmap1[fnum1][hei1+1][wid1+1] = 0;
                }
                    
    }
}

__kernel void returndata(__global int *restrict d_fmap1){
		   int fnum1 = get_global_id(2);
           int hei1 = get_global_id(0);
           int wid1 = get_global_id(1);
int index = wid1 + (hei1 * 34) + (fnum1 * 34 * 34);
d_fmap1[index]=fmap1[index];

}
