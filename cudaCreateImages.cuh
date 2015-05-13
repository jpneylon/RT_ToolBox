
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

typedef unsigned char uchar;


//Input data texture reference
texture<float, 3, cudaReadModeElementType> texImageSrc;
texture<float, 3, cudaReadModeElementType> texOverlaySrc;
texture<float, 3, cudaReadModeElementType> texMaskSrc;


__global__ void
createAxialImage_kernel( uchar *out_data, float min, float max, int3 count, float3 inc, float3 rgb, int scaler, int k, int3 outSize )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= outSize.x || Y >= outSize.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(X) / f_scale;
    float y = __int2float_rn(Y) / f_scale;
    float z = __int2float_rn(k);

    float temp = tex3D(texImageSrc, x+0.5f, y+0.5f, z+0.5f);
    temp -= min;
    temp /= (max - min);

    int p = X + Y * count.x * scaler;

    out_data[3*p + 0]  = (uchar) 255 * temp * rgb.x;
    out_data[3*p + 1]  = (uchar) 255 * temp * rgb.y;
    out_data[3*p + 2]  = (uchar) 255 * temp * rgb.z;
}


__global__ void
createSagittalImage_kernel( uchar *out_data, float min, float max, int3 count, float3 inc, float3 rgb, int scaler, int k, int3 outSize )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= outSize.x || Y >= outSize.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(k);
    float y = (__int2float_rn(X) / f_scale);
    float z = (__int2float_rn(Y) / f_scale);

    float temp = tex3D(texImageSrc, x+0.5f, y+0.5f, z+0.5f);
    temp -= min;
    temp /= (max - min);

    int p = X + Y * count.x * scaler;

    out_data[3*p + 0]  = (uchar) 255 * temp * rgb.x;
    out_data[3*p + 1]  = (uchar) 255 * temp * rgb.y;
    out_data[3*p + 2]  = (uchar) 255 * temp * rgb.z;
}

__global__ void
createOrientationImage_kernel( uchar *out_data, float min, float max, int3 count, float3 inc, float3 rgb, int scaler, int k, int3 outSize )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= outSize.x || Y >= outSize.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(k);
    float y = (__int2float_rn(X) / f_scale) * inc.x / inc.y;
    float z = (__int2float_rn(Y) / f_scale) * inc.x / inc.z;

    float temp = tex3D(texImageSrc, x+0.5f, y+0.5f, z+0.5f);
    temp -= min;
    temp /= (max - min);

    int p = X + Y * count.x * scaler;

    out_data[3*p + 0]  = (uchar) 255 * temp * rgb.x;
    out_data[3*p + 1]  = (uchar) 255 * temp * rgb.y;
    out_data[3*p + 2]  = (uchar) 255 * temp * rgb.z;
}


__global__ void
createCoronalImage_kernel( uchar *out_data, float min, float max, int3 count, float3 inc, float3 rgb, int scaler, int k, int3 outSize )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= outSize.x || Y >= outSize.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(X) / f_scale;
    float y = __int2float_rn(k);
    float z = __int2float_rn(Y) / f_scale;

    float temp = tex3D(texImageSrc, x+0.5f, y+0.5f, z+0.5f);
    temp -= min;
    temp /= (max - min);

    int p = X + Y * count.x * scaler;

    out_data[3*p + 0]  = (uchar) 255 * temp * rgb.x;
    out_data[3*p + 1]  = (uchar) 255 * temp * rgb.y;
    out_data[3*p + 2]  = (uchar) 255 * temp * rgb.z;
}


__device__ float3 doseColorLevel(float t)
{
    if (t > 0.95f )
    {
        float3 red = make_float3(1, 0, 0);
        return(red);
    }
    else if (t > 0.90)
    {
        float3 orange = make_float3(1, 0.64706, 0);
        return(orange);
    }
    else if (t > 0.80)
    {
        float3 yellow = make_float3(1, 1, 0);
        return(yellow);
    }
    else if (t > 0.70)
    {
        float3 green = make_float3(0, 1, 0);
        return(green);
    }
    else if (t > 0.60)
    {
        float3 teal = make_float3(0, 1, 1);
        return(teal);
    }
    else if (t > 0.50)
    {
        float3 blue = make_float3(0, 0, 1);
        return(blue);
    }
    else if (t > 0.25)
    {
        float3 purple = make_float3(0.62745, 0.1255, 0.9412);
        return(purple);
    }
    else
    {
        float3 black = make_float3(0, 0, 0);
        return(black);
    }
}
__device__ float3 doseColorRamp(float t)
{
    const int ncolors = 9;
    float c[ncolors][3] =
    {
        { 0.0, 0.0, 0.0, },
        { 0.1, 0.0, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 1.0, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 0.0, 0.0, },
        { 1.0, 1.0, 1.0, },
    };
    t *= __int2float_rn(ncolors-2);
    int i = __float2int_rd(t);
    float u = t - __int2float_rn(i);
    float3 rgb;
    rgb.x = lerp(c[i][0], c[i+1][0], u);
    rgb.y = lerp(c[i][1], c[i+1][1], u);
    rgb.z = lerp(c[i][2], c[i+1][2], u);

    return(rgb);
}
__global__ void
createDoseImage_kernel( uchar *over_data, float rx_dose, float alpha, int3 count, float3 inc, int scaler, int k )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= scaler*count.x || Y >= scaler*count.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(X) / f_scale;
    float y = __int2float_rn(Y) / f_scale;
    float z = __int2float_rn(k);

    //float mask = floor( 0.5f + tex3D(texMaskSrc, x+0.5f, y+0.5f, z+0.5f) );
    float pixel = tex3D(texOverlaySrc, x+0.5f, y+0.5f, z+0.5f);
    //pixel *= mask;
    pixel /= rx_dose;
    //if (pixel > 1.f) pixel = 1.f;
    //float3 rgb = doseColorRamp(pixel);
    float3 rgb = doseColorLevel(pixel);

    int p = X + Y * count.x * scaler;

    over_data[3*p + 0]  = (uchar) 255*rgb.x;
    over_data[3*p + 1]  = (uchar) 255*rgb.y;
    over_data[3*p + 2]  = (uchar) 255*rgb.z;
}


__device__ float3 gammaColorLevel(float t)
{
    if (t > 3.f )
    {
        float3 red = make_float3(1, 0, 0);
        return(red);
    }
    else if (t > 2.f)
    {
        float3 orange = make_float3(1, 0.64706, 0);
        return(orange);
    }
    else if (t > 1.5f)
    {
        float3 yellow = make_float3(1, 1, 0);
        return(yellow);
    }
    else if (t > 1.f)
    {
        float3 green = make_float3(0, 1, 0);
        return(green);
    }
/*
    else if (t > 0.60)
    {
        float3 teal = make_float3(0, 1, 1);
        return(teal);
    }
    else if (t > 0.50)
    {
        float3 blue = make_float3(0, 0, 1);
        return(blue);
    }
    else if (t > 0.25)
    {
        float3 purple = make_float3(0.62745, 0.1255, 0.9412);
        return(purple);
    }
*/
    else
    {
        float3 black = make_float3(0, 0, 0);
        return(black);
    }
}
__device__ float3 gammaColorRamp(float t)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 0.0, 0.0, 0.0, },
        { 0.0, 1.0, 0.0, },
        { 0.5, 1.0, 0.0, },
        { 1.0, 1.0, 0.0, },
        { 1.0, 0.5, 0.0, },
        { 1.0, 0.0, 0.0, },
        { 1.0, 1.0, 1.0, },
    };
    t *= __int2float_rn(ncolors-2);
    int i = __float2int_rd(t);
    float u = t - __int2float_rn(i);
    float3 rgb;
    rgb.x = lerp(c[i][0], c[i+1][0], u);
    rgb.y = lerp(c[i][1], c[i+1][1], u);
    rgb.z = lerp(c[i][2], c[i+1][2], u);

    return(rgb);
}
__global__ void
createGammaImage_kernel( uchar *over_data, float alpha, int3 count, float3 inc, int scaler, int k )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= scaler*count.x || Y >= scaler*count.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(X) / f_scale;
    float y = __int2float_rn(Y) / f_scale;
    float z = __int2float_rn(k);

    //float mask = floor( 0.5f + tex3D(texMaskSrc, x+0.5f, y+0.5f, z+0.5f) );
    float pixel = tex3D(texOverlaySrc, x+0.5f, y+0.5f, z+0.5f);
    //pixel *= mask;
    //if (pixel < 1.f) pixel = 0.f;
    //pixel /= 3.f;
    //if (pixel > 1.f) pixel = 1.f;
    //float3 rgb = gammaColorRamp(pixel);
    float3 rgb = gammaColorLevel(pixel);

    int p = X + Y * count.x * scaler;

    over_data[3*p + 0]  = (uchar) 255*rgb.x;
    over_data[3*p + 1]  = (uchar) 255*rgb.y;
    over_data[3*p + 2]  = (uchar) 255*rgb.z;
}




__device__ float3 jacobColorRamp(float t)
{
    const int ncolors = 7;
    float c[ncolors][3] =
    {
        { 0.0, 0.0, 0.0, },
        { 0.0, 1.0, 1.0, },
        { 0.0, 0.5, 1.0, },
        { 0.0, 0.0, 1.0, },
        { 0.5, 0.0, 1.0, },
        { 1.0, 0.0, 1.0, },
        { 1.0, 1.0, 1.0, },
    };
    t *= __int2float_rn(ncolors-2);
    int i = __float2int_rd(t);
    float u = t - __int2float_rn(i);
    float3 rgb;
    rgb.x = lerp(c[i][0], c[i+1][0], u);
    rgb.y = lerp(c[i][1], c[i+1][1], u);
    rgb.z = lerp(c[i][2], c[i+1][2], u);

    return(rgb);
}
__global__ void
createJacobImage_kernel( uchar *over_data, float alpha, int3 count, float3 inc, int scaler, int k )
{
    int X = threadIdx.x + blockIdx.x*blockDim.x;
    int Y = threadIdx.y + blockIdx.y*blockDim.y;

    if (X >= scaler*count.x || Y >= scaler*count.y) return;

    float f_scale = __int2float_rn(scaler);
    float x = __int2float_rn(X) / f_scale;
    float y = __int2float_rn(Y) / f_scale;
    float z = __int2float_rn(k);

    float3 rgb;
    //float mask = floor( 0.5f + tex3D(texMaskSrc, x+0.5f, y+0.5f, z+0.5f) );
    float pixel = tex3D(texOverlaySrc, x+0.5f, y+0.5f, z+0.5f);
    //pixel *= mask;
    if (pixel < 0.f)
    {
        pixel *= -1.f;
        if (pixel > 1.f) pixel = 1.f;
        rgb = jacobColorRamp(pixel);
    }
    else
    {
        if (pixel > 1.f) pixel = 1.f;
        rgb = gammaColorRamp(pixel);
    }

    int p = X + Y * count.x * scaler;

    over_data[3*p + 0]  = (uchar) 255*rgb.x;
    over_data[3*p + 1]  = (uchar) 255*rgb.y;
    over_data[3*p + 2]  = (uchar) 255*rgb.z;
}
