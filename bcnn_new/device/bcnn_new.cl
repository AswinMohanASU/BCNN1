#pragma OPENCL EXTENSION cl_altera_channels : enable

channel char c0;
channel char c1;
channel char c2;
channel int lonetoltwo __attribute__((depth(128)));
channel int lonetoltwo1 __attribute__((depth(128)));
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

int count = 0;
while(count < 32*32){
for(fnum = 0; fnum < 128 ; fnum++)
{
act = ((sr[0][0]* d_w1[(3*3*3*fnum)]) +
      (sr[0][1]* d_w1[(3*3*3*fnum)+1]) +
      (sr[0][2]* d_w1[(3*3*3*fnum)+2]) +
      (sr[0][w]* d_w1[(3*3*3*fnum)+3]) +
      (sr[0][(w+1)]* d_w1[(3*3*3*fnum)+4]) +
      (sr[0][(w+2)]* d_w1[(3*3*3*fnum)+5]) +
      (sr[0][2*w]* d_w1[(3*3*3*fnum)+6]) +
      (sr[0][(2*w+1)]* d_w1[(3*3*3*fnum)+7]) +
      (sr[0][(2*w+2)]* d_w1[(3*3*3*fnum)+8]) +
      (sr[1][0]* d_w1[(3*3*3*fnum)+9]) +
      (sr[1][1]* d_w1[(3*3*3*fnum)+10]) +
      (sr[1][2]* d_w1[(3*3*3*fnum)+11]) +
      (sr[1][w]* d_w1[(3*3*3*fnum)+12]) +
      (sr[1][(w+1)]* d_w1[(3*3*3*fnum)+13]) +
      (sr[1][(w+2)]* d_w1[(3*3*3*fnum)+14]) +
      (sr[1][2*w]* d_w1[(3*3*3*fnum)+15]) +
      (sr[1][(2*w+1)]* d_w1[(3*3*3*fnum)+16]) +
      (sr[1][(2*w+2)]* d_w1[(3*3*3*fnum)+17]) +
      (sr[2][0]* d_w1[(3*3*3*fnum)+18]) +
      (sr[2][1]* d_w1[(3*3*3*fnum)+19]) +
      (sr[2][2]* d_w1[(3*3*3*fnum)+20]) +
      (sr[2][w]* d_w1[(3*3*3*fnum)+21]) +
      (sr[2][(w+1)]* d_w1[(3*3*3*fnum)+22]) +
      (sr[2][(w+2)]* d_w1[(3*3*3*fnum)+23]) +
      (sr[2][2*w]* d_w1[(3*3*3*fnum)+24]) +
      (sr[2][(2*w+1)]* d_w1[(3*3*3*fnum)+25]) +
      (sr[2][(2*w+2)]* d_w1[(3*3*3*fnum)+26]));

//if(count >= 32*31 && fnum == 1)
//  printf("%d\n",act); 

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
   for(i1=3; i1 < ((2 * 34) + 3);++i1)
      sr[0][i1-3]=sr[0][i1];
   sr[0][i1-3]=temp[0][0];    
   sr[0][i1-2]=temp[0][1];   
   sr[0][i1-1]=temp[0][2];
   
   for(i2=3; i2 < ((2 * 34) + 3);++i2)
      sr[1][i2-3]=sr[1][i2];
  sr[1][i1-3]=temp[1][0];    
  sr[1][i1-2]=temp[1][1];     
  sr[1][i2-1]=temp[1][2];
   
   for( i3=3; i3 < ((2 * 34) + 3);++i3)
      sr[2][i3-3]=sr[2][i3];
  sr[2][i1-3]=temp[2][0];    
  sr[2][i1-2]=temp[2][1];     
  sr[2][i3-1]=temp[2][2];
}
else if(count%32 != 0 && count%(32*32)!=0){
   for( i1=1; i1 < ((2 * 34) + 3);++i1)
      sr[0][i1-1]=sr[0][i1];
   sr[0][i1-1]=temp[0][0];
   
   for( i2=1; i2 < ((2 * 34) + 3);++i2)
      sr[1][i2-1]=sr[1][i2];
  sr[1][i2-1]=temp[1][0];
   
   for( i3=1; i3 < ((2 * 34) + 3);++i3)
      sr[2][i3-1]=sr[2][i3];
  sr[2][i3-1]=temp[2][0];
}
}

}

//__global char *restrict d_w1,__global char *restrict d_fnum1

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
  //printf("%d\n",temp[0]);
  for(int j=0; j < 128; j++)
    write_channel_altera(lonetoltwo1, temp[j]);
count++;    
}

} 

__kernel void write_data(__global int *restrict d_act1)
{  
for(int i=0; i < 34*34; i++){
  for(int j=0;j<128;j++){
      d_act1[i+((34*34)*j)]=read_channel_altera(lonetoltwo1);
      //printf("index %d value %d\n",i+((32*32)*j),d_act1[i+((32*32)*j)]);
    }
  }
}