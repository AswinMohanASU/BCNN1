#pragma OPENCL EXTENSION cl_altera_channels : enable

channel char c0;
channel char c1;
channel char c2;
channel int lonetoltwo __attribute__((depth(128)));
channel int lonetoltwo1 __attribute__((depth(128)));
channel int lonetoltwo2 __attribute__((depth(128*3)));
channel int ltwotolthree __attribute__((depth(128)));
channel int ltwotolthree1 __attribute__((depth(128)));
channel int ltwotolthree2 __attribute__((depth(128)));

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

__kernel void read_data(__global char *restrict d_fmap0)
{
for(int i=0; i < ((34*34) - 71); i++){
    write_channel_altera(c0, d_fmap0[i+71]);
    write_channel_altera(c1, d_fmap0[i + (34 * 34) + 71]);
    write_channel_altera(c2, d_fmap0[i + (34 * 34 * 2) + 71]);
  }

}

__kernel void layer_one(__global char *restrict d_fmap0,__global char *restrict d_w1,
__global short *restrict d_norm1,
__global bool *restrict d_fmap1
){
int sr[3][((2*34)+3)];
int i,j,i1,i2,i3;
for(i=0; i<((2*34)+3);i++)
{
  sr[0][i]=d_fmap0[i];
  sr[1][i]=d_fmap0[i + (34 * 34)];
  sr[2][i]=d_fmap0[i + (34 * 34 * 2)];
}

int act ;
int w=34;
int fnum;
int fmap1;
int tempchannel[128];
//printf("Start\n"); 
int w1,act_temp;
int count = 0;
while(count < 32*32){
for(fnum = 0; fnum < 128 ; fnum++)
{
  act=0;
 for(int k=0; k < 3; k++)
  {
    w1= ((3*3*3*fnum) + ((k*9)));
    act_temp = ((sr[k][0] * d_w1[w1]) +
      (sr[k][1] * d_w1[(w1+1)]) +
      (sr[k][2] * d_w1[(w1+2)]) +
      (sr[k][w] * d_w1[(w1+3)]) +
      (sr[k][(w+1)] * d_w1[(w1+4)]) +
      (sr[k][(w+2)] * d_w1[(w1+5)]) +
      (sr[k][2*w] * d_w1[(w1+6)]) +
      (sr[k][(2*w+1)] * d_w1[(w1+7)]) +
      (sr[k][(2*w+2)] * d_w1[(w1+8)]));
    act = act + act_temp;

  //if(count == 33 && fnum == 0){  
  //  printf("%d %d %d %d %d %d %d %d %d\n",sr[k][0],sr[k][1],sr[k][2],sr[k][w],sr[k][(w+1)],sr[k][(w+2)],sr[k][2*w],sr[k][((2*w)+1)],sr[k][((2*w)+2)]);
  //
  //  printf("%d %d %d %d %d %d %d %d %d\n",d_w1[w1],d_w1[(w1+1)],d_w1[(w1+2)],d_w1[(w1+3)],d_w1[(w1+4)],d_w1[(w1+5)],d_w1[(w1+6)],d_w1[(w1+7)],d_w1[(w1+8)]);
  //  
  //  printf("%d\n",act);
  //  }
  }
if(act > d_norm1[fnum])
  fmap1 = 1;
else
  fmap1 = 0;
          
tempchannel[fnum]=fmap1;
   
d_fmap1[count+(32*32*fnum)]=fmap1;
}
for(i =0 ; i < 128; i++)
  write_channel_altera(lonetoltwo, tempchannel[i]);

count++;

int temp[3][3],read_flag;
if(count%32 == 0 && count%(32*32)!=0)
  read_flag=3;
else if(count%32 != 0 && count%(32*32)!=0)
  read_flag=1;
else
  read_flag=0;

for(i=0;i<read_flag;i++)
{
  temp[0][i]= read_channel_altera(c0);
}
for(i=0;i<read_flag;i++)
{
  temp[1][i]= read_channel_altera(c1);
}
for(i=0;i<read_flag;i++)
{
  temp[2][i]= read_channel_altera(c2);
}

if(count%32 == 0 && count%(32*32)!=0){
  for(i=0; i < 3; i++) 
  { 
   for(i1=3; i1 < ((2 * 34) + 3);++i1)
      sr[i][i1-3]=sr[i][i1];
    sr[i][i1-3]=temp[i][0];    
    sr[i][i1-2]=temp[i][1];   
    sr[i][i1-1]=temp[i][2];
  }
}
else if(count%32 != 0 && count%(32*32)!=0){
  for(i=0; i < 3 ; i++)
  {
   for( i1=1; i1 < ((2 * 34) + 3);++i1)
      sr[i][i1-1]=sr[i][i1];
    sr[i][i1-1]=temp[i][0];
  } 
 }
}

}

__kernel void add_border(){
int temp[128];
int count=0;
while(count < 34*34){
   for(int i=0; i < 128; i++)
  {
    if(count < 34 || count >= (34*(34-1)) ) 
      temp[i]=0;
    else if(((count%34) == 0) || (((count+1)%34)== 0) )
      temp[i]=0;
    else
      temp[i]=read_channel_altera(lonetoltwo);    
  } 
    if(count < ((3*34)+4))
       for(int j=0; j < 128; j++){
          //write_channel_altera(lonetoltwo2, count);
          write_channel_altera(lonetoltwo2, temp[j]);
        }  
    else
       for(int j=0; j < 128; j++){
          //write_channel_altera(lonetoltwo1, count);
          write_channel_altera(lonetoltwo1, temp[j]);
        }  
count++;    
}

} 


__kernel void layer_two(__global bool *restrict d_w2,
__global short *restrict d_norm2
){
int sr[128][((3*34)+4)];
int i,j,i1,j1,k;
int act1,act2,act3,act4;
int act1_tp,act2_tp,act3_tp,act4_tp;
int w=34;
int fnum;
int fmap2;
int tempchannel[128];
int w1;

int count = 0;
printf("Inside Layer 2\n"); 
for(i=0; i<((3*34)+4);i++)
{
  for(j=0; j < 128; j++)
  {
    sr[j][i]=read_channel_altera(lonetoltwo2);

  }
//  if(i%34==0)
//      printf("\n");
//  printf("%d ",sr[6][i]);
}
printf("after initial read\n");
//for(i=0;i<3*34;i++)
//{
  
//}

while(count < 16*16){
for(fnum = 0; fnum < 128 ; fnum++)
{
  act1=0;
  act2=0;
  act3=0;
  act4=0;
  for(int k=0; k < 128; k++)
  {
      w1= ((128*3*3*fnum) + ((k*9)));
    act1_tp = (!(sr[k][0] ^ d_w2[w1]) +
      !(sr[k][1] ^ d_w2[(w1+1)]) +
      !(sr[k][2] ^ d_w2[(w1+2)]) +
      !(sr[k][w] ^ d_w2[(w1+3)]) +
      !(sr[k][(w+1)] ^ d_w2[(w1+4)]) +
      !(sr[k][(w+2)] ^ d_w2[(w1+5)]) +
      !(sr[k][2*w] ^ d_w2[(w1+6)]) +
      !(sr[k][((2*w)+1)] ^ d_w2[(w1+7)]) +
      !(sr[k][((2*w)+2)] ^ d_w2[(w1+8)]));
    act1 = act1 + act1_tp;
//printf("act1_tp %d\n",act1_tp);
//printf("act1 %d\n",act1);
//printf("%d %d %d %d %d %d %d %d %d\n",sr[k][0],sr[k][1],sr[k][2],sr[k][w],sr[k][(w+1)],sr[k][(w+2)],sr[k][2*w],sr[k][((2*w)+1)],sr[k][((2*w)+2)]);
//printf("%d %d %d %d %d %d %d %d %d\n",d_w2[w1],d_w2[(w1+1)],d_w2[(w1+2)],d_w2[(w1+3)],d_w2[(w1+4)],d_w2[(w1+5)],d_w2[(w1+6)],d_w2[(w1+7)],d_w2[(w1+8)]);
    act2 = act2 + (!(sr[k][1] ^ d_w2[w1]) +
      !(sr[k][2] ^ d_w2[(w1+1)]) +
      !(sr[k][3] ^ d_w2[(w1+2)]) +
      !(sr[k][(w+1)] ^ d_w2[(w1+3)]) +
      !(sr[k][(w+2)] ^ d_w2[(w1+4)]) +
      !(sr[k][(w+3)] ^ d_w2[(w1+5)]) +
      !(sr[k][((2*w)+1)] ^ d_w2[(w1+6)]) +
      !(sr[k][((2*w)+2)] ^ d_w2[(w1+7)]) +
      !(sr[k][((2*w)+3)] ^ d_w2[(w1+8)])) ;

    act3 = act3 + (!(sr[k][w] ^ d_w2[w1]) +
      !(sr[k][(w+1)] ^ d_w2[(w1+1)]) +
      !(sr[k][(w+2)] ^ d_w2[(w1+2)]) +
      !(sr[k][2*w] ^ d_w2[(w1+3)]) +
      !(sr[k][((2*w)+1)] ^ d_w2[(w1+4)]) +
      !(sr[k][((2*w)+2)] ^ d_w2[(w1+5)]) +
      !(sr[k][3*w] ^ d_w2[(w1+6)]) +
      !(sr[k][((3*w)+1)] ^ d_w2[(w1+7)]) +
      !(sr[k][((3*w)+2)] ^ d_w2[(w1+8)])) ;

    act4 = act4 + (!(sr[k][(w+1)] ^ d_w2[w1]) +
      !(sr[k][(w+2)] ^ d_w2[(w1+1)]) +
      !(sr[k][(w+3)] ^ d_w2[(w1+2)]) +
      !(sr[k][((2*w)+1)] ^ d_w2[(w1+3)]) +
      !(sr[k][((2*w)+2)] ^ d_w2[(w1+4)]) +
      !(sr[k][((2*w)+3)] ^ d_w2[(w1+5)]) +
      !(sr[k][((3*w)+1)] ^ d_w2[(w1+6)]) +
      !(sr[k][((3*w)+2)] ^ d_w2[(w1+7)]) +
      !(sr[k][((3*w)+3)] ^ d_w2[(w1+8)])) ;  
if(k<=6 && count==1){   
//printf("3 %d %d %d %d %d %d %d %d %d\n",sr[k][w],sr[k][w+1],sr[k][w+2],sr[k][2*w],sr[k][((2*w)+1)],sr[k][((2*w)+2)],sr[k][3*w],sr[k][((3*w)+1)],sr[k][((3*w)+2)]);
//printf("4 %d %d %d %d %d %d %d %d %d\n",sr[k][w+1],sr[k][w+2],sr[k][w+3],sr[k][((2*w)+1)],sr[k][((2*w)+2)],sr[k][((2*w)+3)],sr[k][((3*w)+1)],sr[k][((3*w)+2)],sr[k][((3*w)+3)]);
//printf("w %d %d %d %d %d %d %d %d %d\n",d_w2[w1],d_w2[(w1+1)],d_w2[(w1+2)],d_w2[(w1+3)],d_w2[(w1+4)],d_w2[(w1+5)],d_w2[(w1+6)],d_w2[(w1+7)],d_w2[(w1+8)]);
//printf("%d %d\n",act3);
}
  }
tempchannel[fnum]=poolnorm(act1,act2,act3,act4,d_norm2[fnum]);      

//if(count%16==0)
//  printf("\n");
//printf(" %d %d",act3,act4);


}

for(i =0 ; i < 128; i++)
  write_channel_altera(ltwotolthree, tempchannel[i]);

count++;

int temp[128][w+4],read_flag;
if(count%16 == 0 && count%(16*16)!=0)
  read_flag=w+4;
else if(count%16 != 0 && count%(16*16)!=0)
  read_flag=2;
else
  read_flag=0;

for(i=0;i<read_flag;i++)
{
  for(j= 0; j<128; j++)
    temp[j][i]= read_channel_altera(lonetoltwo1);
}

if(read_flag==w+4){
  for(i=0; i < 128; i++) 
  { 
   for(i1=38; i1 < ((3 * 34) + 4);++i1) // w+4 => 38 
      sr[i][i1-38]=sr[i][i1];
   for(j1=68; j1 < ((3 * 34) + 4); j1++)  // no.of new data = shiftregister(3W+4) - no.of shift(W+4)
      sr[i][j1]=temp[i][j1-68];  
  }
}
else if(read_flag==2){
  for(i=0; i < 128 ; i++)
  {
   for( i1=2; i1 < ((3 * 34) + 4);++i1)
      sr[i][i1-2]=sr[i][i1];
    sr[i][i1-2]=temp[i][0];
    sr[i][i1-1]=temp[i][1];
  } 
 }

//if(count > 14)
//printf("After 2 shift");
//for(i=0; i< ((3 * 34) + 4); i++ ){
//  if(i%34==0)
//    printf("\n");
//    printf("%d ",sr[6][i]);
//}    
//printf("\n");
}

}

__kernel void add_border_two(){
int temp[128];
int count=0;
printf("Inside add_border_two\n");
while(count < 18*18){
   for(int i=0; i < 128; i++)
  {
    if(count < 18 || count >= (18*(18-1)) ) 
      temp[i]=0;
    else if(((count%18) == 0) || (((count+1)%18)== 0) )
      temp[i]=0;
    else
      temp[i]=read_channel_altera(ltwotolthree);   
  } 
  for(int j=0; j < 128; j++)
    {
     //write_channel_altera(ltwotolthree1, count);
     write_channel_altera(ltwotolthree1, temp[j]);
     }  
count++;    
}

} 

__kernel void write_data(__global int *restrict d_act1)
{  
for(int i=0; i < 18*18; i++){
  for(int j=0;j<128;j++){
      d_act1[i+((18*18)*j)]=read_channel_altera(ltwotolthree1);
      //printf("%d ",d_act1[i+((18*18)*j)]);
    }
  }
}