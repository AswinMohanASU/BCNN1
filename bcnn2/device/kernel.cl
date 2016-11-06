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

__kernel void conv( __global char *restrict d_fmap0,
__global char *restrict d_w1,
__global short *restrict d_norm1,
__global unsigned char *restrict d_dim1,
__global bool *restrict d_fmap1
){

  unsigned char fnum1, hei1, wid1;
  unsigned char i1, j1, k1;
  short temp;
  unsigned int fmap,index;
  unsigned short w;
  local short act1;
           fnum1 = get_global_id(2);
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);

           index = (wid1) + ((hei1) * 34) + (fnum1 * 34 * 34);

if(fnum1 < d_dim1[0] && (hei1 > 0 && hei1 < d_dim1[1]) && (wid1 > 0 && wid1 < d_dim1[2])){
            
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

__kernel void layertwo( __global int *restrict d_fmap1,
__global int *restrict d_w2,
__global int *restrict d_norm2,
__global int *restrict d_act2,
__global int *restrict d_fmap2
){
          int fnum2, hei2, wid2, index,act2_tp, index2;
          int i2, j2, k2, fmap, w;

           fnum2 = get_global_id(2);
           hei2 = get_global_id(0);
           wid2 = get_global_id(1);

           index = (wid2) + ((hei2) * 32) + (fnum2 * 32 * 32);
      for(i2 = 0; i2 < 128; i2++){
          for(j2 = 0; j2 < 3; j2++){
            for(k2 = 0; k2 < 3; k2++){
              
              fmap = ((wid2+k2) + ((hei2+j2) * 34) + (i2 * (34 * 34)));
              
              w = (k2 + (j2 * 3) + (i2 * 3 * 3) + (fnum2 * 3 * 3 * 128));

              act2_tp = !(d_fmap1[fmap] ^ d_w2[w]);

              d_act2[index] = d_act2[index] + act2_tp;

              //printf("index=%d fmap1=%d w2=%d act2_tp=%d d_act2=%d\n",index,d_fmap1[fmap],d_w2[w],act2_tp,d_act2[index]);

            }
          }
        }
        // Max pooling, normalization and non-linearity
        if (hei2%2==1 & wid2%2==1)
        {
          //swap? N
          
          index2 = ((wid2>>1)+1) + (((hei2>>1)+1) * 18) + (fnum2 * 18 * 18);
          
          d_fmap2[index2] = poolnorm(d_act2[index],
                                   d_act2[(wid2) + ((hei2-1) * 32) + (fnum2 * 32 * 32)],
                                   d_act2[(wid2-1) + ((hei2) * 32) + (fnum2 * 32 * 32)],
                                   d_act2[(wid2-1) + ((hei2-1) * 32) + (fnum2 * 32 * 32)],
                                   d_norm2[fnum2]);
        }

}