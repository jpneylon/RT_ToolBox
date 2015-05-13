#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <cuda_runtime_api.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <math.h>

#include <helper_cuda.h>

#include "defs.h"
#include "cudaCreateImages.cuh"



extern "C" void cudaCreateImageSet( FLOAT_GRID *datafloat, IMAGE_SET *datachar, float3 rgb, int3 outSize, int upscaler, uint switcher)
{
    /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
        cudaArray *Vol_Array=0;
		cudaExtent VolumeSize	            =	make_cudaExtent((float)datafloat->size.x, (float)datafloat->size.y, (float)datafloat->size.z);
		cudaChannelFormatDesc channelDesc	=	cudaCreateChannelDesc<float>();
		cudaMalloc3DArray(&Vol_Array, &channelDesc, VolumeSize);

		cudaMemcpy3DParms CopyParams = {0};
		CopyParams.srcPtr	    =	make_cudaPitchedPtr((void*)datafloat->matrix,VolumeSize.width*sizeof(float), VolumeSize.width, VolumeSize.height);
		CopyParams.dstArray	    =	Vol_Array;
		CopyParams.extent	    =	VolumeSize;
		CopyParams.kind		    =	cudaMemcpyHostToDevice;
		cudaMemcpy3D(&CopyParams);

		texImageSrc.normalized		=	false;
		texImageSrc.filterMode		=	cudaFilterModeLinear;
		texImageSrc.addressMode[0]	=	cudaAddressModeClamp;
		texImageSrc.addressMode[1]	=	cudaAddressModeClamp;
		texImageSrc.addressMode[2]	=	cudaAddressModeClamp;

		cudaBindTextureToArray(texImageSrc, Vol_Array, channelDesc);
    //////////////////////////////////////////////////////////////////////////////////////////////

    int imageSize = 3*outSize.x*outSize.y*sizeof( unsigned char );
    uchar *sliceData;
    checkCudaErrors( cudaMalloc( (void**)&sliceData, imageSize ));

    dim3 block(16,16);
    int2 gridSize; gridSize.x = outSize.x/16; gridSize.y = outSize.y/16;
    if ( outSize.x % 16 > 0 ) gridSize.x++;
    if ( outSize.y % 16 > 0 ) gridSize.y++;
    dim3 grid(gridSize.x,gridSize.y);

    for (int k=0; k<outSize.z; k++)
    {
        datachar->anatomy[k].pixels = new unsigned char[3*outSize.x*outSize.y];
        checkCudaErrors(cudaMemset(sliceData,0,imageSize));

        if (switcher == 0)
            createAxialImage_kernel<<<grid,block>>>( sliceData,
                                                     datafloat->min,
                                                     datafloat->max,
                                                     datafloat->size,
                                                     datafloat->voxel,
                                                     rgb,
                                                     upscaler, k,
                                                     outSize );
        else if (switcher == 1)
            createSagittalImage_kernel<<<grid,block>>>( sliceData,
                                                        datafloat->min,
                                                        datafloat->max,
                                                        datafloat->size,
                                                        datafloat->voxel,
                                                        rgb,
                                                        upscaler, k,
                                                        outSize );
        else if (switcher == 2)
            createCoronalImage_kernel<<<grid,block>>>(  sliceData,
                                                        datafloat->min,
                                                        datafloat->max,
                                                        datafloat->size,
                                                        datafloat->voxel,
                                                        rgb,
                                                        upscaler, k,
                                                        outSize );
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");

        checkCudaErrors(cudaMemcpy(datachar->anatomy[k].pixels,sliceData,imageSize,cudaMemcpyDeviceToHost));
        //printf("\n Slice %d created...",k);
    }
    checkCudaErrors(cudaFree(sliceData));
    checkCudaErrors(cudaUnbindTexture(texImageSrc));
    checkCudaErrors(cudaFreeArray(Vol_Array));
}


extern "C" void cudaCreateOrientationImage( FLOAT_GRID *datafloat, IMAGE_SET *datachar, float3 rgb, int3 outSize, int upscaler)
{
    /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
        printf(" Using texture memory...");
        cudaArray *Vol_Array=0;
		cudaExtent VolumeSize	            =	make_cudaExtent((float)datafloat->size.x, (float)datafloat->size.y, (float)datafloat->size.z);
		cudaChannelFormatDesc channelDesc	=	cudaCreateChannelDesc<float>();
		cudaMalloc3DArray(&Vol_Array, &channelDesc, VolumeSize);

		cudaMemcpy3DParms CopyParams = {0};
		CopyParams.srcPtr	    =	make_cudaPitchedPtr((void*)datafloat->matrix,VolumeSize.width*sizeof(float), VolumeSize.width, VolumeSize.height);
		CopyParams.dstArray	    =	Vol_Array;
		CopyParams.extent	    =	VolumeSize;
		CopyParams.kind		    =	cudaMemcpyHostToDevice;
		cudaMemcpy3D(&CopyParams);

		texImageSrc.normalized		=	false;
		texImageSrc.filterMode		=	cudaFilterModeLinear;
		texImageSrc.addressMode[0]	=	cudaAddressModeClamp;
		texImageSrc.addressMode[1]	=	cudaAddressModeClamp;
		texImageSrc.addressMode[2]	=	cudaAddressModeClamp;

		cudaBindTextureToArray(texImageSrc, Vol_Array, channelDesc);
    //////////////////////////////////////////////////////////////////////////////////////////////

    int imageSize = 3*outSize.x*outSize.y*sizeof( uchar );
    uchar *sliceData;
    checkCudaErrors( cudaMalloc( (void**)&sliceData, imageSize ));
    checkCudaErrors( cudaMemset( sliceData, 0, imageSize ) );

    dim3 block(16,16);
    int2 gridSize; gridSize.x = outSize.x/16; gridSize.y = outSize.y/16;
    if ( outSize.x % 16 > 0 ) gridSize.x++;
    if ( outSize.y % 16 > 0 ) gridSize.y++;
    dim3 grid(gridSize.x,gridSize.y);

    createOrientationImage_kernel<<<grid,block>>>(  sliceData,
                                                    datafloat->min,
                                                    datafloat->max,
                                                    datafloat->size,
                                                    datafloat->voxel,
                                                    rgb,
                                                    upscaler, datafloat->size.x/2,
                                                    outSize );
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

    datachar->anatomy[0].pixels = new unsigned char[3*outSize.x*outSize.y];
    checkCudaErrors(cudaMemcpy(datachar->anatomy[0].pixels,sliceData,imageSize,cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(sliceData));
    checkCudaErrors(cudaUnbindTexture(texImageSrc));
    checkCudaErrors(cudaFreeArray(Vol_Array));
}


extern "C" void cudaCreateOverlayImageSet( FLOAT_GRID *overlay, IMAGE_SET *datachar, float RX_DOSE, float alpha, int upscaler, int switcher)
{
    /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
        printf("\n Using texture memory...\n"); fflush(stdout);
        cudaArray *Over_Array;
		cudaExtent VolumeSize	            =	make_cudaExtent((float)overlay->size.x, (float)overlay->size.y, (float)overlay->size.z);
		cudaChannelFormatDesc channelDesc	=	cudaCreateChannelDesc<float>();
		cudaMalloc3DArray(&Over_Array, &channelDesc, VolumeSize);

		cudaMemcpy3DParms CopyParamsOver = {0};
		CopyParamsOver.srcPtr	    =	make_cudaPitchedPtr((void*)overlay->matrix,VolumeSize.width*sizeof(float), VolumeSize.width, VolumeSize.height);
		CopyParamsOver.dstArray	    =	Over_Array;
		CopyParamsOver.extent	    =	VolumeSize;
		CopyParamsOver.kind		    =	cudaMemcpyHostToDevice;
		cudaMemcpy3D(&CopyParamsOver);

		texOverlaySrc.normalized		=	false;
		texOverlaySrc.filterMode		=	cudaFilterModeLinear;
		texOverlaySrc.addressMode[0]	=	cudaAddressModeClamp;
		texOverlaySrc.addressMode[1]	=	cudaAddressModeClamp;
		texOverlaySrc.addressMode[2]	=	cudaAddressModeClamp;

		cudaBindTextureToArray(texOverlaySrc, Over_Array, channelDesc);
    //////////////////////////////////////////////////////////////////////////////////////////////

    int X = upscaler*overlay->size.x;
    int Y = upscaler*overlay->size.y;
    int Z = overlay->size.z;

    int imageSize = 3*X*Y*sizeof( uchar );
    printf("\n IMAGE SIZE: 3 x %d x %d x %lu = %d", X, Y, sizeof(uchar), imageSize);

    uchar *sliceData;
    checkCudaErrors( cudaMalloc( (void**)&sliceData, imageSize ));

    dim3 block(16,16);
    int2 gridSize; gridSize.x = X/16; gridSize.y = Y/16;
    if ( X%16 > 0 ) gridSize.x++;
    if ( Y%16 > 0 ) gridSize.y++;
    dim3 grid(gridSize.x,gridSize.y);

    for (int k=0; k<Z; k++)
    {
        checkCudaErrors(cudaMemset(sliceData,0,imageSize));

        if (switcher == 0){
            createDoseImage_kernel<<<grid,block>>>( sliceData,
                                                    RX_DOSE,
                                                    alpha,
                                                    overlay->size,
                                                    overlay->voxel,
                                                    upscaler, k );
        }
        else if (switcher == 1){
            createGammaImage_kernel<<<grid,block>>>(sliceData,
                                                    alpha,
                                                    overlay->size,
                                                    overlay->voxel,
                                                    upscaler, k );
        }
        else if (switcher == 2){
            createJacobImage_kernel<<<grid,block>>>(sliceData,
                                                    alpha,
                                                    overlay->size,
                                                    overlay->voxel,
                                                    upscaler, k );
        }
        cudaDeviceSynchronize();
        getLastCudaError("Kernel execution failed");

        datachar->overlay[k].pixels = new unsigned char[3*X*Y];
        checkCudaErrors(cudaMemcpy(datachar->overlay[k].pixels,sliceData,imageSize,cudaMemcpyDeviceToHost));
    }

    checkCudaErrors(cudaFree(sliceData));
    checkCudaErrors(cudaUnbindTexture(texOverlaySrc));
    checkCudaErrors(cudaFreeArray(Over_Array));
}





