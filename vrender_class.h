#ifndef __VRENDER_CLASS_H__
#define __VRENDER_CLASS_H__

// CUDA utilities and system includes
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>

// CUDA Includes
#include <vector_functions.h>
#include <driver_functions.h>

#define PI 3.1416

typedef struct
{
    float4 m[3];
} float3x4;
struct Ray
{
    float3 o;	// origin
    float3 d;	// direction
};


class VRender
{
public:
    VRender()
    {
        width = 512;
        height = 512;
        blockSize.x = 16;
        blockSize.y = 16;
        viewRotation = make_float3(90,180,0);
        viewTranslation = make_float3(0,0,3);
        density = 0.1;
        brightness = 1.5;
        transferOffset = 0;
        transferScale = 1;
        weight = 1;
        volumeSize = make_cudaExtent(256,256,256);
    }

    ~VRender()
    {
        cleanup_vrender();
    }



    int init_vrender( float *data,
                      int3  data_size,
                      float data_max,
                      float data_min,
                      int colorScale );
    void set_vrender_parameters( float r_dens, float r_bright, float r_offset, float r_scale );
    void set_vrender_rotation( float dx, float dy );
    void set_vrender_translation( float dx, float dy );
    void set_vrender_zoom ( float dy );

    unsigned char *get_vrender_buffer();

    int get_width()
    {
        return width;
    };
    int get_height()
    {
        return height;
    };
    float get_density()
    {
        return density;
    };
    float get_brightness()
    {
        return brightness;
    };
    float get_offset()
    {
        return transferOffset;
    };
    float get_scale()
    {
        return transferScale;
    };
    float get_last_x()
    {
        return last_x;
    };
    float get_last_y()
    {
        return last_y;
    };

    void set_width( int i )
    {
        width = i;
    };
    void set_height( int i )
    {
        height = i;
    };
    void set_density( float v )
    {
        density = v;
    };
    void set_brightness( float v )
    {
        brightness = v;
    };
    void set_offset( float v )
    {
        transferOffset = v;
    };
    void set_scale( float v )
    {
        transferScale = v;
    };
    void set_last_x( float v )
    {
        last_x = v;
    };
    void set_last_y( float v )
    {
        last_y = v;
    };


private:

    int iDivUp(int a, int b)
    {
        return (a % b != 0) ? (a / b + 1) : (a / b);
    }

    void printModelViewMatrix();
    void setInvViewMatrix();
    void render();
    void translateMat( float *matrix, float3 translation );
    void rotMat( float *matrix, float3 axis, float theta, float3 center );
    void multiplyModelViewMatrix( float *trans );
    void transformModelViewMatrix();
    int  cleanup_vrender();

    void initDeviceCharVolume(float *data,
                              float data_max,
                              float data_min );
    void initRenderArrays(int colorScale);
    void freeCudaBuffers();
    void render_kernel(dim3 , dim3 , unsigned char *, uint , uint ,
                       float , float , float , float , float );
    void copyInvViewMatrix(float *, size_t );

    cudaExtent volumeSize;

    uint width;
    uint height;

    dim3 blockSize;
    dim3 gridSize;

    float3 viewRotation;
    float3 viewTranslation;

    float invViewMatrix[12];
    float identityMatrix[16];
    float modelViewMatrix[16];

    float density;
    float brightness;
    float transferOffset;
    float transferScale;
    float weight;
    float last_x;
    float last_y;

    int3 vsize;
    unsigned char *render_buf;

    unsigned char *d_charvol;
    unsigned char *d_output;
};

#endif // __VRENDER_CLASS_H__

