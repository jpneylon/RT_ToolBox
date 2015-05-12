#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>





//Input data texture reference
texture<float, 3, cudaReadModeElementType> texData;
texture<float, 3, cudaReadModeElementType> texDose;


__global__ void
cudaResampleXY( float *output, float3 outVox, float3 inVox, float3 offset, int3 inCount )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int pos = i + j*blockDim.x + k*blockDim.x*gridDim.x;
    if (pos >= gridDim.x*gridDim.y*blockDim.x) return;

    float xstep = (__int2float_rn(i) * outVox.x - offset.x) / inVox.x;
    float ystep = (__int2float_rn(j) * outVox.y - offset.y) / inVox.y;
    float zstep = __int2float_rn(k);

    if (xstep < 0.f || ystep < 0.f || zstep < 0.f ||
        xstep >= inCount.x-1 || ystep >= inCount.y-1 || zstep >= inCount.z-1 )
            output[pos] = 0.f;
    else
            output[pos] = tex3D(texData,xstep+0.5f,ystep+0.5f,zstep+0.5f);
}


__global__ void
cudaResampleZ( float *output, float outVox, float inVox, float offset, int inCount )
{
    int i = threadIdx.x;
    int j = blockIdx.x;
    int k = blockIdx.y;
    int pos = i + j*blockDim.x + k*blockDim.x*gridDim.x;
    if (pos >= gridDim.x*gridDim.y*blockDim.x) return;

    float xstep = __int2float_rn(i);
    float ystep = __int2float_rn(j);
    float zstep = (__int2float_rn(k) * outVox - offset) / inVox;

    if (zstep < 0.f || zstep >= inCount-1 )
        output[pos] = 0.f;
    else
        output[pos] = tex3D(texData,xstep+0.5f,ystep+0.5f,zstep+0.5f);
}
