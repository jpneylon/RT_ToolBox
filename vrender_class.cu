
#include "vrender_class.h"

typedef unsigned char uchar;

__constant__ float3x4 c_invViewMatrix;
__constant__ int dID;
__constant__ float dmax, dmin;

texture<unsigned char, 3, cudaReadModeNormalizedFloat> tex;
texture<float4, 1, cudaReadModeElementType> transferTex;

cudaArray *d_volumeArray;
cudaArray *d_transferFuncArray;

cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
cudaMemcpy3DParms copyParams = {0};


__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

	*tnear = largest_tmin;
	*tfar = smallest_tmax;

	return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 get_eye_ray_direction(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 get_eye_ray_origin(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

__device__ float4
get_pix_val( int src, int maxSteps, Ray eyeRay, float tstep, float tnear, float tfar,
		     float Offset, float Scale, float dens, float weight, float opacity )
{
    float t = tnear;
    float3 pos = eyeRay.o + eyeRay.d*tnear;
    float3 step = eyeRay.d*tstep;
    float4 sum = make_float4(0.0f);

    for(int i=0; i<maxSteps; i++)
    {
        // read from 3D texture
        // remap position to [0, 1] coordinates
        float sample;
        float4 col;

        if (src == 1){
                sample = tex3D(tex, pos.x*0.5f+0.5f, pos.y*0.5f+0.5f, pos.z*0.5f+0.5f);
                col = tex1D(transferTex, (sample-Offset)*Scale);
        }

        // lookup in transfer function texture
        col.w *= dens * weight;

        // pre-multiply alpha
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;
        // "over" operator for front-to-back blending
        sum = sum + col*(1.0f - sum.w);

        t += tstep;
        if (t > tfar) break;

        pos += step;
    }

    return sum;
}

__global__ void
d_render(unsigned char *d_output, uint imageW, uint imageH,
         float dens, float bright,
         float Offset, float Scale, float weight)
{
    const int maxSteps = 500;
    const float tstep = 0.01f;
    const float opacityThreshold = 0.95f;
    const float3 boxMin = make_float3(-1.0f, -1.0f, -1.0f);
    const float3 boxMax = make_float3(1.0f, 1.0f, 1.0f);

	uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;
    if ((x >= imageW) || (y >= imageH)) return;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3( get_eye_ray_origin(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)) );
    eyeRay.d = normalize(make_float3(u, v, 2.0f));
    eyeRay.d = get_eye_ray_direction(c_invViewMatrix, eyeRay.d);

    // find intersection with box
	float tnear, tfar;
	int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
    if (!hit) return;
	if (tnear < 0.0f) tnear = 0.0f;     // clamp to near plane

    // march along ray from front to back, accumulating color
    float4 sum = make_float4(0.0f);

    sum += get_pix_val( 1, maxSteps, eyeRay, tstep, tnear, tfar, Offset, Scale, dens, weight, opacityThreshold );
    sum *= bright;

    // write output color
    float4 rgba;
    rgba.x = __saturatef(sum.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(sum.y);
    rgba.z = __saturatef(sum.z);
    rgba.w = __saturatef(sum.w);
    d_output[3*(x + imageW * y) + 0] = (unsigned char) (255 * rgba.x);
    d_output[3*(x + imageW * y) + 1] = (unsigned char) (255 * rgba.y);
    d_output[3*(x + imageW * y) + 2] = (unsigned char) (255 * rgba.z);
}

__global__ void
deviceDub2Char( float *input, unsigned char *out )
{
    int pos = threadIdx.x + blockIdx.x*blockDim.x + blockIdx.y*gridDim.x*blockDim.x;
    float value = 255 * (input[pos] - dmin) / abs(dmax - dmin);
    out[pos] = (unsigned char) value;
}



void
VRender::initDeviceCharVolume(  float *data,
                                float data_max,
                                float data_min )
{
    int Xc = volumeSize.width;
    int Yc = volumeSize.height;
    int Zc = volumeSize.depth;
    size_t float_size = Xc*Yc*Zc*sizeof(float);

    float *ddata;
    checkCudaErrors( cudaMalloc( (void **) &ddata, float_size) );
    checkCudaErrors( cudaMemset( ddata, 0.0, float_size ) );

    dim3 block(Xc);
    dim3 grid(Yc,Zc);
    size_t data_size = Xc*Yc*Zc*sizeof(unsigned char);

    checkCudaErrors( cudaMalloc( (void**) &d_charvol, data_size ) );
    checkCudaErrors( cudaMemset( d_charvol, 0.0, data_size ) );

    checkCudaErrors( cudaMemcpy( ddata, data, float_size, cudaMemcpyHostToDevice ) );
    //checkCudaErrors( cudaHostAlloc( (void**) &vdens, data_size, cudaHostAllocPortable ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmax, &data_max, sizeof(float) ) );
    checkCudaErrors( cudaMemcpyToSymbol( dmin, &data_min, sizeof(float) ) );

    deviceDub2Char<<<grid,block>>>(ddata,d_charvol);
    cudaThreadSynchronize();

    checkCudaErrors(cudaFree(ddata));
}

void
VRender::initRenderArrays( int colorScale )
{
    checkCudaErrors( cudaMalloc( (void**) &d_output, width * height * 3 * sizeof(uchar) ) );
    checkCudaErrors( cudaMemset( d_output, 0, width * height * 3 * sizeof(uchar) ) );

    // create 3D array
    checkCudaErrors( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

    // copy data to 3D array
    copyParams.srcPtr   = make_cudaPitchedPtr(d_charvol, volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
    copyParams.dstArray = d_volumeArray;
    copyParams.extent   = volumeSize;
    copyParams.kind     = cudaMemcpyDeviceToDevice;
    checkCudaErrors( cudaMemcpy3D(&copyParams) );

    // set texture parameters
    tex.normalized = true;                      // access with normalized texture coordinates
    tex.filterMode = cudaFilterModeLinear;      // linear interpolation
    tex.addressMode[0] = cudaAddressModeClamp;  // clamp texture coordinates
    tex.addressMode[1] = cudaAddressModeClamp;

    // bind array to 3D texture
    checkCudaErrors(cudaBindTextureToArray(tex, d_volumeArray, channelDesc));

    // create transfer function texture
    float4 transferFunc3[] = {
        {  0.5, 0.0, 0.5, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.5, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.5, 1.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 1.0, 0.5, 1.0, },
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.5, 1.0, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  1.0, 0.825, 0.0, 1.0, },
        {  1.0, 0.65, 0.0, 1.0, },
        {  1.0, 0.325, 0.0, 1.0, },
        {  1.0, 0.0, 0.0, 1.0  },
        {  1.0, 0.5, 0.5, 1.0  },
    };

    float4 transferFunc2[] = {
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.5, 0.0, 0.5, 1.0, },
        {  1.0, 0.0, 1.0, 1.0, },
        {  0.5, 0.0, 1.0, 1.0, },
        {  0.0, 0.0, 1.0, 1.0, },
        {  0.0, 0.5, 1.0, 1.0, },
        {  0.0, 1.0, 1.0, 1.0, },
        {  0.0, 1.0, 0.5, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.0, 1.0, 0.0, 1.0, },
        {  0.5, 1.0, 0.0, 1.0, },
        {  1.0, 1.0, 0.0, 1.0, },
        {  1.0, 0.825, 0.0, 1.0, },
        {  1.0, 0.65, 0.0, 1.0, },
        {  1.0, 0.325, 0.0, 1.0, },
        {  1.0, 0.0, 0.0, 1.0  },
        {  1.0, 0.5, 0.5, 1.0  },
        {  1.0, 1.0, 1.0, 1.0  }
    };

    float4 transferFunc1[] = {
        {  0.0, 0.0, 0.0, 0.0  },
        {  0.2, 0.2, 0.2, 1.0, },
        {  0.4, 0.4, 0.4, 1.0, },
        {  0.6, 0.6, 0.6, 1.0, },
        {  0.8, 0.8, 0.8, 1.0, },
        {  1.0, 1.0, 1.0, 1.0, }
    };

    cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();
    cudaArray* d_transferFuncArray;

    if (colorScale <= 1)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc1)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc1, sizeof(transferFunc1), cudaMemcpyHostToDevice));
    }
    else if (colorScale == 2)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc2)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc2, sizeof(transferFunc2), cudaMemcpyHostToDevice));
    }
    else if (colorScale == 3)
    {
        checkCudaErrors(cudaMallocArray( &d_transferFuncArray, &channelDesc2, sizeof(transferFunc3)/sizeof(float4), 1));
        checkCudaErrors(cudaMemcpyToArray( d_transferFuncArray, 0, 0, transferFunc3, sizeof(transferFunc3), cudaMemcpyHostToDevice));
    }

    transferTex.filterMode = cudaFilterModeLinear;
    transferTex.normalized = true;    // access with normalized texture coordinates
    transferTex.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

    // Bind the array to the texture
    checkCudaErrors( cudaBindTextureToArray( transferTex, d_transferFuncArray, channelDesc2));
}

void
VRender::freeCudaBuffers()
{
    checkCudaErrors(cudaUnbindTexture(tex));
    checkCudaErrors(cudaUnbindTexture(transferTex));
    cudaFreeArray(d_transferFuncArray);
    checkCudaErrors(cudaFree(d_charvol));
}

void
VRender::render_kernel(dim3 gridSize, dim3 blockSize, unsigned char *d_output, uint imageW, uint imageH,
                            float dens, float bright, float Offset, float Scale, float weight)
{
    d_render<<<gridSize, blockSize>>>( d_output, imageW, imageH, dens, bright,
                                        Offset, Scale, weight );
}

void
VRender::copyInvViewMatrix(float *invViewMatrix, size_t sizeofMatrix)
{
    checkCudaErrors( cudaMemcpyToSymbol(c_invViewMatrix, invViewMatrix, sizeofMatrix, 0, cudaMemcpyHostToDevice) );
}


void
VRender::printModelViewMatrix()
{
    printf("\n Model View Matrix:");
    printf("\n  %2.3f %2.3f %2.3f %2.3f",modelViewMatrix[0],
                                         modelViewMatrix[1],
                                         modelViewMatrix[2],
                                         modelViewMatrix[3]);
    printf("\n  %2.3f %2.3f %2.3f %2.3f",modelViewMatrix[4],
                                         modelViewMatrix[5],
                                         modelViewMatrix[6],
                                         modelViewMatrix[7]);
    printf("\n  %2.3f %2.3f %2.3f %2.3f",modelViewMatrix[8],
                                         modelViewMatrix[9],
                                         modelViewMatrix[10],
                                         modelViewMatrix[11]);
    printf("\n  %2.3f %2.3f %2.3f %2.3f",modelViewMatrix[12],
                                         modelViewMatrix[13],
                                         modelViewMatrix[14],
                                         modelViewMatrix[15]);
    printf("\n");
}

void
VRender::setInvViewMatrix()
{
    invViewMatrix[0]  = modelViewMatrix[0];     //rot_x.x
    invViewMatrix[1]  = modelViewMatrix[1];     //rot_x.y
    invViewMatrix[2]  = modelViewMatrix[2];     //rot_x.z
    invViewMatrix[3]  = modelViewMatrix[3];     //trans.x
    invViewMatrix[4]  = modelViewMatrix[4];     //rot_y.x
    invViewMatrix[5]  = modelViewMatrix[5];     //rot_y.y
    invViewMatrix[6]  = modelViewMatrix[6];     //rot_y.z
    invViewMatrix[7]  = modelViewMatrix[7];     //trans.y
    invViewMatrix[8]  = modelViewMatrix[8];     //rot_z.x
    invViewMatrix[9]  = modelViewMatrix[9];     //rot_z.y
    invViewMatrix[10] = modelViewMatrix[10];    //rot_z.z
    invViewMatrix[11] = modelViewMatrix[11];    //trans.z

    copyInvViewMatrix(invViewMatrix, sizeof(float4)*3);
}

// render image using CUDA
void
VRender::render( )
{
    // clear image
    checkCudaErrors( cudaMemset( d_output, 0, width * height * 3 * sizeof(unsigned char) ) );
    // call CUDA kernel, writing results to PBO
    render_kernel(gridSize, blockSize, d_output, width, height, density, brightness, transferOffset, transferScale, weight);
    getLastCudaError("Kernel execution failed");
}

unsigned char *
VRender::get_vrender_buffer()
{
    render();
    checkCudaErrors( cudaMemcpy( render_buf, d_output, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost ) );
    return render_buf;
}

void
VRender::translateMat( float *matrix, float3 translation )
{
    float3 start = make_float3( matrix[3],
                                matrix[7],
                                matrix[11]);

    start.x -= translation.x;
    start.y -= translation.y;
    start.z -= translation.z;

    float3 new_origin;
    new_origin.x = matrix[0] * start.x + matrix[1] * start.y + matrix[2]  * start.z;
    new_origin.y = matrix[4] * start.x + matrix[5] * start.y + matrix[6]  * start.z;
    new_origin.z = matrix[8] * start.x + matrix[9] * start.y + matrix[10] * start.z;

    matrix[3]  = new_origin.x;
    matrix[7]  = new_origin.y;
    matrix[11] = new_origin.z;
}

void
VRender::rotMat( float *matrix, float3 axis, float theta, float3 center ) // theta in degrees
{
    for (int v=0; v<3; v++)
    {
        float3 rot = make_float3( (theta * PI / 180) * axis.x,
                                  (theta * PI / 180) * axis.y,
                                  (theta * PI / 180) * axis.z);

        float3 start = make_float3( matrix[0 + v*4] - center.x,
                                    matrix[1 + v*4] - center.y,
                                    matrix[2 + v*4] - center.z);

        float3 inter_x = make_float3( start.x,
                                      start.y * cos(rot.x) - start.z * sin(rot.x),
                                      start.z * cos(rot.x) + start.y * sin(rot.x) );

        float3 inter_y = make_float3( inter_x.x * cos(rot.y) + inter_x.z * sin(rot.y),
                                      inter_x.y,
                                      inter_x.z * cos(rot.y) - inter_x.x * sin(rot.y) );

        float3 inter_z = make_float3( inter_y.x * cos(rot.z) - inter_y.y * sin(rot.z),
                                      inter_y.y * cos(rot.z) + inter_y.x * sin(rot.z),
                                      inter_y.z );

        matrix[0 + v*4] = inter_z.x + center.x;
        matrix[1 + v*4] = inter_z.y + center.y;
        matrix[2 + v*4] = inter_z.z + center.z;
    }
}

void
VRender::multiplyModelViewMatrix( float *trans )
{
    float *result;
    result = new float[16];

    for (int r=0; r<3; r++)
    for (int c=0; c<3; c++)
    {

        float4 row = make_float4( trans[0 + 4*r],
                                  trans[1 + 4*r],
                                  trans[2 + 4*r],
                                  trans[3 + 4*r] );

        float4 col = make_float4( modelViewMatrix[0  + c],
                                  modelViewMatrix[4  + c],
                                  modelViewMatrix[8  + c],
                                  modelViewMatrix[12 + c] );

        result[c + 4*r] = row.x * col.x +
                          row.y * col.y +
                          row.z * col.z +
                          row.w * col.w;
    }

    memcpy( modelViewMatrix, result, 16*sizeof(float) );
    delete [] result;
}

void
VRender::transformModelViewMatrix()
{
    float *matrix;
    matrix = new float[16];
    memcpy( matrix, identityMatrix, 16*sizeof(float) );
    rotMat( matrix, make_float3(1,0,0), -viewRotation.x, make_float3(0,0,0));
    rotMat( matrix, make_float3(0,1,0), -viewRotation.y, make_float3(0,0,0));
    translateMat( matrix, viewTranslation );

    //multiplyModelViewMatrix( matrix );
    memcpy( modelViewMatrix, matrix, 16*sizeof(float) );

    delete [] matrix;
}


void
VRender::set_vrender_rotation( float dx, float dy )
{
    viewRotation.x -= dy;
    viewRotation.y += dx;

    transformModelViewMatrix();
    setInvViewMatrix();
}

void
VRender::set_vrender_translation( float dx, float dy )
{
    viewTranslation.x += dx;
    viewTranslation.y -= dy;

    transformModelViewMatrix();
    setInvViewMatrix();
}

void
VRender::set_vrender_zoom ( float dy )
{
    viewTranslation.z += dy;

    transformModelViewMatrix();
    setInvViewMatrix();
}

int
VRender::init_vrender(float *data,
                      int3  data_size,
                      float data_max,
                      float data_min,
                      int colorScale )
{
    volumeSize.width  = data_size.x;
    volumeSize.height = data_size.y;
    volumeSize.depth  = data_size.z;

    render_buf = new unsigned char[ height * width * 3 ];
    memset( render_buf, 0, height * width * 3 );

    initDeviceCharVolume( data, data_max, data_min );
    initRenderArrays( colorScale );

    // calculate new grid size
    gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

    memset(identityMatrix, 0, 16*sizeof(float));
    identityMatrix[0] = 1;
    identityMatrix[5] = 1;
    identityMatrix[10] = 1;
    identityMatrix[15] = 1;

    memcpy(modelViewMatrix, identityMatrix, 16*sizeof(float));
    //printModelViewMatrix();

    transformModelViewMatrix();
    //printModelViewMatrix();

    setInvViewMatrix();
    //printf("\n density: %2.3f | brightness: %2.3f | offset: %2.3f | scale: %2.3f",density,brightness,transferOffset,transferScale);

    return 1;
}

int
VRender::cleanup_vrender()
{
    delete [] render_buf;
    checkCudaErrors( cudaFree(d_output) );
    freeCudaBuffers();
    return 1;
}

