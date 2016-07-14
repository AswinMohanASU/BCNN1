__global int fmap1 [128][34][34];

// 1st layer

__global int act1[128][32][32];
__global int temp[128][32][32];
__kernel void intialize(){

		   int fnum1 = get_global_id(0);
           int hei1 = get_global_id(1);
           int wid1 = get_global_id(2);
		   
		   fmap1[fnum1][hei1][wid1]= 0;
		   if(hei1 < 32 && wid1 < 32)
		  		act1[fnum1][hei1][wid1]=0;
}

__kernel void Conv1 (__global int *fmap0,
__global int *w1,
__global int *norm1,
__global int *restrict x
){

	int fnum1, hei1, wid1;
	int i1, j1, k1,f,w;

     	   fnum1 = get_global_id(0);
           hei1 = get_global_id(1);
           wid1 = get_global_id(2);
 
 	   
 for(i1 = 0; i1 < 3; i1++){
 	for(j1 = 0; j1 < 3; j1++){
 		for(k1 = 0; k1 < 3; k1++){
 				f = ((wid1+k1) + ((hei1+j1) * 34) + (i1 * (34 * 34)));
 				w = k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3);
 				temp[fnum1][hei1][wid1] = fmap0[f] * w1[w];
 				act1[fnum1][hei1][wid1] = (act1[fnum1][hei1][wid1] + temp[fnum1][hei1][wid1]) ;

            }
        }
    }
    
    // Normalization and non-linearity
    if(act1[fnum1][hei1][wid1] > norm1[fnum1])
        fmap1[fnum1][hei1+1][wid1+1] = 1;
    else
        fmap1[fnum1][hei1+1][wid1+1] = 0;


if(fnum1 < 3)	
	x[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))] = fmap0[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))];

}

__kernel void returndata(__global int *restrict y, __global int *restrict z){

		   int fnum1 = get_global_id(0);
           int hei1 = get_global_id(1);
           int wid1 = get_global_id(2);
y[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))]= fmap1[fnum1][hei1][wid1];
if (wid1 < 34 && hei1 < 34)
	z[(wid1 + (hei1 * 34) + (fnum1 * (34 * 34)))]= temp[fnum1][hei1][wid1];

}