#define int12 0xFFF
#define int2 0x3
#define int6 0x3F
#define int1 0x1
#define int13 0x1FFF
#define int14 0x3FFF
#define int32 0xFFFFFFFF

bool poolnorm (ushort a1, ushort a2, ushort a3, ushort a4, ushort norm){
  ushort s1,s2,s3;
  bool dout;

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

__kernel void layerone( __global char *restrict d_fmap0,
__global char *restrict d_w1,
__global short *restrict d_norm1,
__global uchar *restrict d_dim1,
__global bool *restrict d_fmap1
){

  uchar fnum1, hei1, wid1;
  int i1, j1, k1, fmap, w, index;
  short temp;
  short act1;
           fnum1 = get_global_id(2);
           hei1 = get_global_id(0);
           wid1 = get_global_id(1);

           index = (wid1) + ((hei1) * 34) + (fnum1 * 34 * 34);

if(fnum1 < d_dim1[0] && (hei1 > 0 && hei1 < d_dim1[1]) && (wid1 > 0 && wid1 < d_dim1[2])){
            
            act1 = 0;
            #pragma unroll 1
            for(i1 = 0; i1 < 3; i1++){
                #pragma unroll 3
                for(j1 = 0; j1 < 3; j1++){
                    #pragma unroll 3
                    for(k1 = 0; k1 < 3; k1++){

                            fmap = ((wid1+k1-1) + ((hei1+j1-1) * 34) + (i1 * (34 * 34)));
                            w = (k1 + (j1 * 3) + (i1 * 3 * 3) + (fnum1 * 3 * 3 * 3));
		      printf("w = %d\n",w);	
                            temp =  d_fmap0[fmap] * d_w1[w];
                            act1 =  act1 + temp;

                        }
                    }
                }

               // Normalization and non-linearity
                if ( act1 >  d_norm1[fnum1])
                    d_fmap1[index] = 1;
                else
                    d_fmap1[index] = 0;
    }
    else
        d_fmap1[index] = 0;

//printf("index=%d fmap=%d fmap_mask=%d\n",index,d_fmap0[fmap],(0x3F & d_fmap0[fmap]));
      
}

__kernel void layertwo( __global bool *restrict d_fmap1,
__global bool *restrict d_w2,
__global ushort *restrict d_norm2,
__global ushort *restrict d_act2,
__global bool *restrict d_fmap2
){
          ushort act2;
          uchar fnum2, hei2, wid2,i2, j2, k2;
          int index,index2,fmap, w;
          bool act2_tp;

           fnum2 = get_global_id(2);
           hei2 = get_global_id(0);
           wid2 = get_global_id(1);

           index = (wid2) + ((hei2) * 32) + (fnum2 * 32 * 32);

           act2 = 0;
      for(i2 = 0; i2 < 128; i2++){
          #pragma unroll 3
          for(j2 = 0; j2 < 3; j2++){
            #pragma unroll 3
            for(k2 = 0; k2 < 3; k2++){
              
              fmap = ((wid2+k2) + ((hei2+j2) * 34) + (i2 * (34 * 34)));
              
              w = (k2 + (j2 * 3) + (i2 * 3 * 3) + (fnum2 * 3 * 3 * 128));

              act2_tp = int1 & !((int1 & d_fmap1[fmap]) ^ (int1 & d_w2[w]));

              act2 = int12 & ((int12 & act2) + (int1 & act2_tp));
            }
          }
        }

        d_act2[index] = int12 & act2;
        // Max pooling, normalization and non-linearity
        
        if(wid2 < 18 && hei2 < 18){
          d_fmap2[(wid2) + ((hei2) * 18) + (fnum2 * 18 * 18)] = 0;
        }
        
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
__kernel void layerthree( __global bool *restrict d_fmap2,
__global bool *restrict d_w3,
__global ushort *restrict d_norm3,
__global ushort *restrict d_dim3,
__global ushort *restrict d_act3,
__global bool *restrict d_fmap3
){

  ushort fnum3, hei3, wid3;
  ushort i3, j3, k3;
  int index,index2,fmap,w;
  ushort act3;
  bool temp;
           fnum3 = get_global_id(2);
           hei3 = get_global_id(0);
           wid3 = get_global_id(1);

           index = (wid3) + ((hei3) * 18) + (fnum3 * 18 * 18);

if(fnum3 < d_dim3[0] && (hei3 > 0 && hei3 < d_dim3[1]) && (wid3 > 0 && wid3 < d_dim3[2])){
            index2 = (wid3-1) + ((hei3-1) * 16) + (fnum3 * 16 * 16);
            act3 = 0;
            for(i3 = 0; i3 < 128; i3++){
               #pragma unroll 3
                for(j3 = 0; j3 < 3; j3++){
                    #pragma unroll 3
                    for(k3 = 0; k3 < 3; k3++){

                            fmap = ((wid3+k3-1) + ((hei3+j3-1) * 18) + (i3 * (18 * 18)));
                            w = (k3 + (j3 * 3) + (i3 * 3 * 3) + (fnum3 * 3 * 3 * 128));

                            temp = int1 &  !((int1 & d_fmap2[fmap]) ^ (int1 & d_w3[w]));

                            act3 = ((int12 & act3) + (int1 & temp));

                        }
                    }
                }
                d_act3[index2]= int12 & act3;
                // Normalization and non-linearity

                if((int12 & act3) > (int12 & d_norm3[fnum3]))
                    d_fmap3[index] = 1;
                else
                    d_fmap3[index] = 0;    
    }
    else
        d_fmap3[index] = 0;
    
} 
__kernel void layerfour( __global bool *restrict d_fmap3,
__global bool *restrict d_w4,
__global ushort *restrict d_norm4,
__global ushort *restrict d_act4,
__global bool *restrict d_fmap4
){
          ushort act4;
          ushort fnum4, hei4, wid4; 
          int index,index2, fmap, w;
          ushort i4, j4, k4;
          bool act4_tp;
           fnum4 = get_global_id(2);
           hei4 = get_global_id(0);
           wid4 = get_global_id(1);

           index = (wid4) + ((hei4) * 16) + (fnum4 * 16 * 16);

           act4 = 0;
      for(i4 = 0; i4 < 256; i4++){
          #pragma unroll 3
          for(j4 = 0; j4 < 3; j4++){
            #pragma unroll 3
            for(k4 = 0; k4 < 3; k4++){
              
              fmap = ((wid4+k4) + ((hei4+j4) * 18) + (i4 * (18 * 18)));
              
              w = (k4 + (j4 * 3) + (i4 * 3 * 3) + (fnum4 * 3 * 3 * 256));

              act4_tp = int1 & !((int1 & d_fmap3[fmap]) ^ (int1 & d_w4[w]));

              act4 = int12 & ((int12 & act4) + (int1 & act4_tp));
            }
          }
        }

        d_act4[index] = int12 & act4;
        // Max pooling, normalization and non-linearity
        
        if(wid4 < 10 && hei4 < 10){
          d_fmap4[(wid4) + ((hei4) * 10) + (fnum4 * 10 * 10)] = 0;
        }
        
        if (hei4%2==1 & wid4%2==1)
        {
          //swap? N
          
          index2 = ((wid4>>1)+1) + (((hei4>>1)+1) * 10) + (fnum4 * 10 * 10);
          
          d_fmap4[index2] = poolnorm(d_act4[index],
                                   d_act4[(wid4) + ((hei4-1) * 16) + (fnum4 * 16 * 16)],
                                   d_act4[(wid4-1) + ((hei4) * 16) + (fnum4 * 16 * 16)],
                                   d_act4[(wid4-1) + ((hei4-1) * 16) + (fnum4 * 16 * 16)],
                                   d_norm4[fnum4]);         
        }

}

__kernel void layerfive( __global bool *restrict d_fmap4,
__global bool *restrict d_w5,
__global ushort *restrict d_norm5,
__global ushort *restrict d_dim5,
__global ushort *restrict d_act5,
__global bool *restrict d_fmap5
){
  ushort fnum5, hei5, wid5;
  ushort i5, j5, k5;
  int fmap, w, index,index2;
  ushort act5;
  bool temp;
           fnum5 = get_global_id(2);
           hei5 = get_global_id(0);
           wid5 = get_global_id(1);

           index = (wid5) + ((hei5) * 10) + (fnum5 * 10 * 10);

if(fnum5 < d_dim5[0] && (hei5 > 0 && hei5 < d_dim5[1]) && (wid5 > 0 && wid5 < d_dim5[2])){

            index2 = (wid5-1) + ((hei5-1) * 8) + (fnum5 * 8 * 8);
            act5 = 0;
            for(i5 = 0; i5 < 256; i5++){
                #pragma unroll 3
                for(j5 = 0; j5 < 3; j5++){
                    #pragma unroll 3
                    for(k5 = 0; k5 < 3; k5++){

                            fmap = ((wid5+k5-1) + ((hei5+j5-1) * 10) + (i5 * (10 * 10)));
                            
                            w = (k5 + (j5 * 3) + (i5 * 3 * 3) + (fnum5 * 3 * 3 * 256));

                            temp =int1 & !((int1 & d_fmap4[fmap]) ^ (int1 & d_w5[w]));

                            act5 = int12 & ((int12 & act5) + (int1 & temp));
                        }
                    }
                }
                d_act5[index2]=int12 & act5;
                // Normalization and non-linearity

                if((int12 & act5) > (int12 & d_norm5[fnum5]))
                    d_fmap5[index] = 1;
                else
                    d_fmap5[index] = 0;    
    }
    else
        d_fmap5[index] = 0;
    
}
__kernel void layersix( __global bool *restrict d_fmap5,
__global bool *restrict d_w6,
__global ushort *restrict d_norm6,
__global ushort *restrict d_act6,
__global bool *restrict d_fmap6
){
          ushort act6;
          ushort fnum6, hei6, wid6;
          int index,index2,fmap,w;
          ushort i6, j6, k6;
          bool act6_tp;

           fnum6 = get_global_id(2);
           hei6 = get_global_id(0);
           wid6 = get_global_id(1);

           index = ((wid6 + ((hei6) * 8) + (fnum6 * 8 * 8)));

           act6 = 0;
      for(i6= 0; i6 < 512; i6++){
          #pragma unroll 3
          for(j6 = 0; j6 < 3; j6++){
            #pragma unroll 3
            for(k6 = 0; k6 < 3; k6++){
              
              fmap = ((wid6+k6) + ((hei6+j6) * 10) + (i6 * (10 * 10)));
              
              w = (k6 + (j6 * 3) + (i6 * 3 * 3) + (fnum6 * 3 * 3 * 512));

              act6_tp =int1 & !((int1 & d_fmap5[fmap]) ^ (int1 & d_w6[w]));

              act6 = ((int14 & act6) + (int1 & act6_tp));
            }
          }
        }

        d_act6[index] = int14 & act6;
        // Max pooling, normalization and non-linearity
        
        if(wid6 < 4 && hei6 < 4){
          d_fmap6[(wid6) + ((hei6) * 4) + (fnum6 * 4 * 4)] = 0;
        }
        
        if (hei6%2==1 & wid6%2==1)
        {
          //swap? N
          
          index2 = (fnum6<<4)+(((hei6>>1))<<2)+(wid6>>1);
          
          d_fmap6[index2] = poolnorm(d_act6[index],
                                   d_act6[(wid6) + ((hei6-1) * 8) + (fnum6 * 8 * 8)],
                                   d_act6[(wid6-1) + ((hei6) * 8) + (fnum6 * 8 * 8)],
                                   d_act6[(wid6-1) + ((hei6-1) * 8) + (fnum6 * 8 * 8)],
                                   d_norm6[fnum6]);         
        }

}
__kernel void layerseven( __global bool *restrict d_fmap6,
__global bool *restrict d_w7,
__global ushort *restrict d_norm7,
__global ushort *restrict d_act7,
__global bool *restrict d_fmap7
){
    ushort j7, k7;
    int w;
    bool act7_tp;
    ushort act7;

           j7 = get_global_id(0);
    act7 = 0;       
    for(k7 = 0; k7 < 8192; k7++){
      w = j7 + (k7 * 1024);
      
      act7_tp = int1 & !((int1 & d_fmap6[k7]) ^ (int1 & d_w7[w]));
      
      act7 = int14 &  ((int14 & act7) + (int1 & act7_tp));
    }
    
    d_act7[j7]=int14 & act7;
    
    // Normalization and non-linearity
    if(act7 > d_norm7[j7])
      d_fmap7[j7] = 1;
    else
      d_fmap7[j7] = 0;  
}
__kernel void layereight( __global bool *restrict d_fmap7,
__global bool *restrict d_w8,
__global ushort *restrict d_norm8,
__global ushort *restrict d_act8,
__global bool *restrict d_fmap8
){
    ushort j8, k8;
    bool act8_tp;
    ushort act8;
    
    j8 = get_global_id(0);

    for(k8 = 0; k8 < 1024; k8++){
      act8_tp = int1 & !((int1 & d_fmap7[k8]) ^ (int1 & d_w8[(j8 + (k8 * 1024))]));
      act8= int12 & ((int12 & act8) + (int1 & act8_tp));
    }
    
    d_act8[j8]=int12 & act8;
    
    // Normalization and non-linearity
    if(act8 > d_norm8[j8])
      d_fmap8[j8] = 1;
    else
      d_fmap8[j8] = 0;


}


__kernel void layernine( __global bool *restrict d_fmap8,
__global bool *restrict d_w9,
__global ushort *restrict d_norm9,
__global ushort *restrict d_act9,
__global ushort *restrict d_fmap9
){
    ushort j9, k9;
    bool act9_tp;
    ushort act9;
    
           j9 = get_global_id(0);

    for(k9 = 0; k9 < 1024; k9++){
      act9_tp =int1 & !((int1 & d_fmap8[k9]) ^ (int1 & d_w9[j9 + (k9 * 10)]));
      act9= int12 & ((int12 & act9) + (int1 & act9_tp));
    }
    
    d_act9[j9]=int12 & act9;
    
    // Normalization
    d_fmap9[j9] = int12 & ((int12 & act9) - (int12 & d_norm9[j9]));
}
