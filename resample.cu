
#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <sys/stat.h>

#include "defs.h"

#include "resample.cuh"



extern "C" void resampleXY( FLOAT_GRID *hin,
                            FLOAT_GRID *hout,
                            float *output )
{
    printf("\nResampling in XY plane on GPU...\n");

    const int dataSize = hout->size.x * hout->size.y * hout->size.z * sizeof(float);
    float *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, dataSize));

    /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
    printf(" Using texture memory...");
    cudaArray *Vol_Array=0;
    cudaExtent VolumeSize	=	make_cudaExtent((float)hin->size.x, (float)hin->size.y, (float)hin->size.z);
    cudaChannelFormatDesc channelDesc	=	cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&Vol_Array, &channelDesc, VolumeSize);

    cudaMemcpy3DParms CopyParams = {0};
    CopyParams.srcPtr	    =	make_cudaPitchedPtr((void*)hin->matrix,VolumeSize.width*sizeof(float), VolumeSize.width, VolumeSize.height);
    CopyParams.dstArray	    =	Vol_Array;
    CopyParams.extent       =	VolumeSize;
    CopyParams.kind		    =	cudaMemcpyHostToDevice;
    cudaMemcpy3D(&CopyParams);

    texData.normalized		=	false;
    texData.filterMode		=	cudaFilterModeLinear;
    texData.addressMode[0]	=	cudaAddressModeClamp;
    texData.addressMode[1]	=	cudaAddressModeClamp;
    texData.addressMode[2]	=	cudaAddressModeClamp;

    cudaBindTextureToArray(texData, Vol_Array, channelDesc);
    //////////////////////////////////////////////////////////////////////////////////////////////

    dim3 grid(hout->size.y, hout->size.z);
    dim3 block(hout->size.x);

    printf("\n grid: (%d,%d) \t block: %d \n",grid.x,grid.y,block.x);

    cudaResampleXY<<<grid,block>>>( d_out, hout->voxel, hin->voxel, hout->offset, hin->size );
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

    checkCudaErrors(cudaMemcpy(output, d_out, dataSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaUnbindTexture(texData));
    checkCudaErrors(cudaFreeArray(Vol_Array));
}


extern "C" void resampleZ( FLOAT_GRID *hin,
                           FLOAT_GRID *hout,
                           float *input, float oldSliceThickness )
{
    printf("\nResampling in Axial plane on GPU...\n");
    const int outSize = hout->size.x * hout->size.y * hout->size.z * sizeof(float);
    float *d_out;
    checkCudaErrors(cudaMalloc((void **)&d_out, outSize));

    /////////////////// Bind Inputs to 3D Texture Arrays //////////////////////////////////////////////
    printf(" Using texture memory...");
    cudaArray *Vol_Array=0;
    cudaExtent VolumeSize	            =	make_cudaExtent((float)hout->size.x, (float)hout->size.y, (float)hin->size.z);
    cudaChannelFormatDesc channelDesc	=	cudaCreateChannelDesc<float>();
    cudaMalloc3DArray(&Vol_Array, &channelDesc, VolumeSize);

    cudaMemcpy3DParms CopyParams = {0};
    CopyParams.srcPtr	    =	make_cudaPitchedPtr((void*)input,VolumeSize.width*sizeof(float), VolumeSize.width, VolumeSize.height);
    CopyParams.dstArray	    =	Vol_Array;
    CopyParams.extent	    =	VolumeSize;
    CopyParams.kind		    =	cudaMemcpyHostToDevice;
    cudaMemcpy3D(&CopyParams);

    texData.normalized		=	false;
    texData.filterMode		=	cudaFilterModeLinear;
    texData.addressMode[0]	=	cudaAddressModeClamp;
    texData.addressMode[1]	=	cudaAddressModeClamp;
    texData.addressMode[2]	=	cudaAddressModeClamp;

    cudaBindTextureToArray(texData, Vol_Array, channelDesc);
    //////////////////////////////////////////////////////////////////////////////////////////////

    dim3 grid(hout->size.y, hout->size.z);
    dim3 block(hout->size.x);

    cudaResampleZ<<<grid,block>>>( d_out, hout->voxel.z, oldSliceThickness, hout->offset.z, hin->size.z );
    cudaDeviceSynchronize();
    getLastCudaError("Kernel execution failed");

    printf("\n grid: (%d,%d) \t block: %d \n",grid.x,grid.y,block.x);

    checkCudaErrors(cudaMemcpy(hout->matrix, d_out, outSize, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_out));
    checkCudaErrors(cudaUnbindTexture(texData));
    checkCudaErrors(cudaFreeArray(Vol_Array));
}


extern "C" void cuda_doseResize( FLOAT_GRID *kVin, FLOAT_GRID *Dose )
{
    printf("\n Dimensions of Data Set 1: %d x %d x %d",kVin->size.x,kVin->size.y,kVin->size.z);
    printf("\n Voxel Size of Data Set 1: %2.2f x %2.2f x %2.2f",kVin->voxel.x,kVin->voxel.y,kVin->voxel.z);
    printf("\n MAX = %3.2f \t MIN = %3.2f",kVin->max,kVin->min);

    printf("\n Dimensions of Data Set 2: %d x %d x %d",Dose->size.x,Dose->size.y,Dose->size.z);
    printf("\n Voxel Size of Data Set 2: %2.2f x %2.2f x %2.2f",Dose->voxel.x,Dose->voxel.y,Dose->voxel.z);
    printf("\n MAX = %3.2f \t MIN = %3.2f",Dose->max,Dose->min);

    printf("\n Position for Data 1: (%2.2f,%2.2f,%2.2f)",kVin->startPos.x,kVin->startPos.y,kVin->startPos.z);
    printf("\n Position for Data 2: (%2.2f,%2.2f,%2.2f)\n\n",Dose->startPos.x,Dose->startPos.y,Dose->startPos.z);

    FLOAT_GRID *tempDose;
    tempDose = new FLOAT_GRID;

    tempDose->voxel.x = kVin->voxel.x;
    tempDose->voxel.y = kVin->voxel.y;
    tempDose->voxel.z = Dose->voxel.z;
    printf("\n Voxel Size of Output2: %2.2f x %2.2f x %2.2f\n",tempDose->voxel.x,tempDose->voxel.y,tempDose->voxel.z);

    tempDose->size.x = kVin->size.x;
    tempDose->size.y = kVin->size.y;
    tempDose->size.z = Dose->size.z;
    printf("\n Dimensions of Output2: %d x %d x %d\n",tempDose->size.x,tempDose->size.y,tempDose->size.z);

    float r1 = kVin->startPos.x - Dose->startPos.x;
    tempDose->offset.x = abs(r1);
    if (Dose->size.x*Dose->voxel.x > kVin->size.x*kVin->voxel.x) tempDose->offset.x *= -1.f;

    float c1 = kVin->startPos.y - Dose->startPos.y;
    tempDose->offset.y = abs(c1);
    if (Dose->size.y*Dose->voxel.y > kVin->size.y*kVin->voxel.y) tempDose->offset.y *= -1.f;

    float b1 = abs(Dose->startPos.z - kVin->startPos.z);
    tempDose->offset.z = abs(b1);
    if (Dose->size.z*Dose->voxel.z > kVin->size.z*kVin->voxel.z) tempDose->offset.z *= -1.f;

    printf("\n Offset for Data 2: (%2.2f,%2.2f,%2.2f)\n",tempDose->offset.x,tempDose->offset.y,tempDose->offset.z);

    float *mv_tempmat;
    int moutSize = tempDose->size.x*tempDose->size.y*tempDose->size.z*sizeof(float);
    mv_tempmat = (float*)malloc( moutSize );

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);
    resampleXY( Dose,
                tempDose,
                mv_tempmat );
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    float cudatime;
    cudaEventElapsedTime(&cudatime,start,stop);
    printf("\n\n X-Y Resampling time: %4.3f msec\n",cudatime);

    delete []Dose->matrix;
    tempDose->voxel.z = kVin->voxel.z;
    tempDose->size.z = kVin->size.z;
    int outSize = tempDose->size.x*tempDose->size.y*tempDose->size.z*sizeof(float);

    Dose->matrix = (float*)malloc(outSize);
    tempDose->matrix = (float*)malloc(outSize);
    memset(tempDose->matrix,0.0f,outSize);

    cudaEventRecord(start,0);
    resampleZ( Dose, tempDose,
               mv_tempmat, Dose->voxel.z );
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cudatime,start,stop);
    printf("\n\n Z Resampling time: %4.3f msec\n",cudatime);

    memcpy(Dose->matrix,tempDose->matrix,outSize);
    free(tempDose->matrix);
    free(mv_tempmat);

    Dose->size.x = kVin->size.x;
    Dose->size.y = kVin->size.y;
    Dose->size.z = kVin->size.z;
    Dose->voxel.x = kVin->voxel.x;
    Dose->voxel.y = kVin->voxel.y;
    Dose->voxel.z = kVin->voxel.z;
    Dose->startPos.x = kVin->startPos.x;
    Dose->startPos.y = kVin->startPos.y;
    Dose->startPos.z = kVin->startPos.z;

    delete tempDose;
    printf("Finished.\n");
}


int resize_dose_data( FLOAT_GRID *kVin, FLOAT_GRID *dose_in )
{
    clock_t start = clock();

    cudaDeviceReset();
    cudaSetDevice(GPU);
    cuda_doseResize( kVin, dose_in );

    printf("\n Total time elapsed: %4.3f msec\n\n\n", ((float)clock() - start) * 1000 / CLOCKS_PER_SEC );

    return(1);
}
