#define HAVE_CONFIG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <gtk/gtk.h>
#include <cuda_runtime_api.h>


#include "defs.h"

extern "C" void cudaCreateImageSet( FLOAT_GRID *, IMAGE_SET *, float3, int3, int, uint );
extern "C" void cudaCreateOrientationImage( FLOAT_GRID *, IMAGE_SET *, float3, int3, int );
extern "C" void cudaCreateOverlayImageSet( FLOAT_GRID *, IMAGE_SET *, float, float, int, int );


void createAxialImageSet( FLOAT_GRID *datafloat, IMAGE_SET *axial_out, float3 rgb, int upscaler )
{
    int3 outSize;
    outSize.x = upscaler*datafloat->size.x;
    outSize.y = upscaler*datafloat->size.y;
    outSize.z = datafloat->size.z;

    cudaCreateImageSet( datafloat, axial_out, rgb, outSize, upscaler, 0 );
}
void createSagittalImageSet(FLOAT_GRID *datafloat, IMAGE_SET *sagittal_out, float3 rgb, int upscaler )
{
    int3 outSize;
    outSize.x = upscaler*datafloat->size.y;
    outSize.y = upscaler*datafloat->size.z;
    outSize.z = datafloat->size.x;

    cudaCreateImageSet( datafloat, sagittal_out, rgb, outSize, upscaler, 1 );
}
void createCoronalImageSet(FLOAT_GRID *datafloat, IMAGE_SET *coronal_out, float3 rgb, int upscaler )
{
    int3 outSize;
    outSize.x = upscaler*datafloat->size.x;
    outSize.y = upscaler*datafloat->size.z;
    outSize.z = datafloat->size.y;

    cudaCreateImageSet( datafloat, coronal_out, rgb, outSize, upscaler, 2 );
}
void createOverlayImageSet( FLOAT_GRID *overlay, IMAGE_SET *datachar, float color_scale_max, int upscaler, int OVERLAY_TYPE )
{
    float alpha = 0.95;

    cudaCreateOverlayImageSet( overlay, datachar, color_scale_max, alpha, upscaler, OVERLAY_TYPE );
}
void createOrientationImage(FLOAT_GRID *datafloat, IMAGE_SET *sagittal_out, float3 rgb, int upscaler )
{
    int3 outSize;
    outSize.x = upscaler * floor(0.5f + (float)datafloat->size.y * datafloat->voxel.y / datafloat->voxel.x);
    outSize.y = upscaler * floor(0.5f + (float)datafloat->size.z * datafloat->voxel.z / datafloat->voxel.x);
    outSize.z = 1;

    cudaCreateOrientationImage( datafloat, sagittal_out, rgb, outSize, upscaler );
}


int add_structure_contours( FLOAT_GRID *ctdata, IMAGE_SET *images, CNTR_SPECS *object, int upscaler )
{
    //printf("\n Contour %d = %s",object->ROInumber,object->ROIname);
    //printf("\n   Contour points: %d",object->TOTpoints);
    //printf("\n   Contour color: (%d, %d, %d)",object->rgb.x,object->rgb.y,object->rgb.z);
    //printf("\n   Contour pos: (%3.2f, %3.2f, %3.2f)\n",object->matrix[0],object->matrix[1],object->matrix[2]);
    //printf("\n   ct pos: (%3.2f, %3.2f, %3.2f)\n",ctdata->startPos.x,ctdata->startPos.y,ctdata->startPos.z);
    //printf("\n   ct inc: (%3.2f, %3.2f, %3.2f)\n",ctdata->voxel.x,ctdata->voxel.y,ctdata->voxel.z);
    //printf("\n   ct start: (%3.2f, %3.2f, %3.2f)\n",ctdata->start.z,ctdata->start.y,ctdata->start.z);

    int3 last;
    last.x = 0;
    last.y = 0;
    last.z = 0;
    int subtotal = 0;

    for (int c=0; c<object->subCntrs; c++)
    {
        //printf("\n C = %d | subtotal = %d\n",object->CTRpoints[c],subtotal);
        for (int p=0; p<object->CTRpoints[c]; p++)
        {
            float x = (object->matrix[3*(p + subtotal)] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
            float y = (object->matrix[3*(p + subtotal)+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
            float z = (ctdata->startPos.z - object->matrix[3*(p + subtotal)+2]) / ctdata->voxel.z;

            int3 coord;
            coord.x = (int)floor(x + 0.5f);
            coord.y = (int)floor(y + 0.5f);
            coord.z = (int)floor(z + 0.5f);
            //printf("   %d  %d  %d\n",coord.x,coord.y,coord.z); fflush(stdout);

            int vox = coord.x + coord.y * upscaler * ctdata->size.x;
            images->anatomy[coord.z].pixels[3*vox + 0]  = (unsigned char) object->rgb.x;
            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) object->rgb.y;
            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) object->rgb.z;
        }
        subtotal += object->CTRpoints[c];
    }

    subtotal = 0;
    bool newSlice = true;
    for (int c=0; c<object->subCntrs; c++)
    {
        for (int p=0; p<=object->CTRpoints[c]; p++)
        {
            float x,y,z;
            if (p==object->CTRpoints[c])
            {
                x = (object->matrix[3*subtotal] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
                y = (object->matrix[3*subtotal+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
                z = (ctdata->startPos.z - object->matrix[3*subtotal+2]) / ctdata->voxel.z;
            }
            else
            {
                x = (object->matrix[3*(p + subtotal)] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
                y = (object->matrix[3*(p + subtotal)+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
                z = (ctdata->startPos.z - object->matrix[3*(p + subtotal)+2]) / ctdata->voxel.z;
            }

            int3 coord;
            coord.x = (int)floor(x + 0.5f);
            coord.y = (int)floor(y + 0.5f);
            coord.z = (int)floor(z + 0.5f);

            if (newSlice)
            {
                newSlice = false;
            }
            else
            {
                int2 diff, step, dir;
                diff.x = coord.x - last.x;
                diff.y = coord.y - last.y;
                dir.x = (diff.x > 0) ? 1 : -1;
                dir.y = (diff.y > 0) ? 1 : -1;
                step.x = last.x;
                step.y = last.y;
                bool stepSwitch = abs(diff.x) > abs(diff.y);
                while( step.x != coord.x || step.y != coord.y)
                {
                    if (stepSwitch)
                    {
                        if(step.x != coord.x)
                        {
                            step.x += dir.x;
                            int vox = step.x + step.y * upscaler * ctdata->size.x;
                            images->anatomy[coord.z].pixels[3*vox]  = (unsigned char) object->rgb.x;
                            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) object->rgb.y;
                            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) object->rgb.z;
                        }
                        stepSwitch = !stepSwitch;
                    }
                    else
                    {
                        if(step.y != coord.y)
                        {
                            step.y += dir.y;
                            int vox = step.x + step.y * upscaler * ctdata->size.x;
                            images->anatomy[coord.z].pixels[3*vox]  = (unsigned char) object->rgb.x;
                            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) object->rgb.y;
                            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) object->rgb.z;
                        }
                        stepSwitch = !stepSwitch;
                    }
                }
            }
            last.x = coord.x;
            last.y = coord.y;
            last.z = coord.z;
        }
        newSlice = true;
        subtotal += object->CTRpoints[c];
    }

    return(1);
}
int remove_structure_contours( FLOAT_GRID *ctdata, IMAGE_SET *images, CNTR_SPECS *object, int upscaler )
{
    int3 last;
    last.x = 0;
    last.y = 0;
    last.z = 0;
    int subtotal = 0;

    for (int c=0; c<object->subCntrs; c++)
    {
        for (int p=0; p<object->CTRpoints[c]; p++)
        {
            float x = (object->matrix[3*(p + subtotal)] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
            float y = (object->matrix[3*(p + subtotal)+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
            float z = (ctdata->startPos.z - object->matrix[3*(p + subtotal)+2]) / ctdata->voxel.z;

            int3 coord;
            coord.x = (int)floor(x + 0.5f);
            coord.y = (int)floor(y + 0.5f);
            coord.z = (int)floor(z + 0.5f);

            float value = GRID_VALUE(ctdata,(coord.x/upscaler),(coord.y/upscaler),coord.z);

            int vox = coord.x + coord.y * upscaler * ctdata->size.x;
            images->anatomy[coord.z].pixels[3*vox]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
        }
        subtotal += object->CTRpoints[c];
    }

    subtotal = 0;
    bool newSlice = true;
    for (int c=0; c<object->subCntrs; c++)
    {
        for (int p=0; p<object->CTRpoints[c]; p++)
        {
            float x,y,z;
            if (p==object->CTRpoints[c])
            {
                x = (object->matrix[3*subtotal] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
                y = (object->matrix[3*subtotal+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
                z = (ctdata->startPos.z - object->matrix[3*subtotal+2]) / ctdata->voxel.z;
            }
            else
            {
                x = (object->matrix[3*(p + subtotal)] - ctdata->startPos.x) / (ctdata->voxel.x / upscaler);
                y = (object->matrix[3*(p + subtotal)+1] - ctdata->startPos.y) / (ctdata->voxel.y / upscaler);
                z = (ctdata->startPos.z - object->matrix[3*(p + subtotal)+2]) / ctdata->voxel.z;
            }

            int3 coord;
            coord.x = (int)floor(x + 0.5f);
            coord.y = (int)floor(y + 0.5f);
            coord.z = (int)floor(z + 0.5f);

            if (newSlice)
            {
                newSlice = false;
            }
            else
            {
                int2 diff, step, dir;
                diff.x = coord.x - last.x;
                diff.y = coord.y - last.y;
                dir.x = (diff.x > 0) ? 1 : -1;
                dir.y = (diff.y > 0) ? 1 : -1;
                step.x = last.x;
                step.y = last.y;
                bool stepSwitch = abs(diff.x) > abs(diff.y);
                while( step.x != coord.x || step.y != coord.y)
                {
                    if (stepSwitch)
                    {
                        if(step.x != coord.x)
                        {
                            step.x += dir.x;

                            float value = GRID_VALUE(ctdata,(step.x/upscaler),(step.y/upscaler),coord.z);

                            int vox = step.x + step.y * upscaler * ctdata->size.x;
                            images->anatomy[coord.z].pixels[3*vox]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                        }
                        stepSwitch = !stepSwitch;
                    }
                    else
                    {
                        if(step.y != coord.y)
                        {
                            step.y += dir.y;

                            float value = GRID_VALUE(ctdata,(step.x/upscaler),(step.y/upscaler),coord.z);

                            int vox = step.x + step.y * upscaler * ctdata->size.x;
                            images->anatomy[coord.z].pixels[3*vox]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[coord.z].pixels[3*vox + 1]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[coord.z].pixels[3*vox + 2]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                        }
                        stepSwitch = !stepSwitch;
                    }
                }
            }
            last.x = coord.x;
            last.y = coord.y;
            last.z = coord.z;
        }
        newSlice = true;
        subtotal += object->CTRpoints[c];
    }

    return(1);
}
int add_warped_contours( FLOAT_GRID *ctdata, IMAGE_SET *images, CNTR_SPECS *object, FLOAT_GRID *volume, int upscaler )
{
    //printf("\n Contour %d = %s",object->ROInumber,object->ROIname);
    //printf("\n   Contour color: (%d, %d, %d)",object->rgb.x,object->rgb.y,object->rgb.z);

    float frac =1.f;

    for (int k=0; k<ctdata->size.z; k++)
        for (int j=0; j<ctdata->size.y; j++)
            for (int i=0; i<ctdata->size.x; i++)
            {
                int pos = i + ctdata->size.x *(j + ctdata->size.y*k);
                if ( volume->matrix[pos] > 0.5f )
                {
                    float value = 255 * (GRID_VALUE(ctdata,i,j,k) - ctdata->min) / (ctdata->max - ctdata->min);
                    float red, green, blue;
                    red = (1.f - frac)*value + frac * (float)object->rgb.x;
                    blue = (1.f - frac)*value + frac * (float)object->rgb.y;
                    green = (1.f - frac)*value + frac * (float)object->rgb.z;

                    for (int x=0; x<upscaler; x++)
                        for (int y=0; y<upscaler; y++)
                        {
                            int p = (i*upscaler + x) + (j*upscaler + y) * upscaler * ctdata->size.x;
                            images->anatomy[k].pixels[3*p + 0]  = (unsigned char) red; //object->rgb.x;
                            images->anatomy[k].pixels[3*p + 1]  = (unsigned char) blue; //object->rgb.y;
                            images->anatomy[k].pixels[3*p + 2]  = (unsigned char) green; //object->rgb.z;
                        }
                }
            }

    return(1);
}
int remove_warped_contours( FLOAT_GRID *ctdata, IMAGE_SET *images, CNTR_SPECS *object, FLOAT_GRID *volume, int upscaler )
{
    for (int k=0; k<ctdata->size.z; k++)
        for (int j=0; j<ctdata->size.y; j++)
            for (int i=0; i<ctdata->size.x; i++)
            {
                int pos = i + ctdata->size.x *(j + ctdata->size.y*k);
                if ( volume->matrix[pos] > 0.5f )
                {
                    float value = GRID_VALUE(ctdata,i,j,k);

                    for (int x=0; x<upscaler; x++)
                        for (int y=0; y<upscaler; y++)
                        {
                            int p = (i*upscaler + x) + (j*upscaler + y) * upscaler * ctdata->size.x;
                            images->anatomy[k].pixels[3*p + 0]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[k].pixels[3*p + 1]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                            images->anatomy[k].pixels[3*p + 2]  = (unsigned char) 255 * (value - ctdata->min) / (ctdata->max - ctdata->min);
                        }
                }
            }

    return(1);
}


int add_crosshairs( FLOAT_GRID *dens, IMAGE_SET *images, float3 pos, float red, float green, float blue, float alpha, int upscaler, int crossSize )
{
    int2 crosshair;
    crosshair.x = upscaler * (int)floor(pos.x + 0.5f);
    crosshair.y = upscaler * (int)floor(pos.y + 0.5f);

    int X = upscaler*dens->size.x;
    int Y = upscaler*dens->size.y;
    int slice = (int)floor(pos.z + 0.5f);

    float temp;
    for (int x=crosshair.x-crossSize; x<crosshair.x+crossSize+1; x++)
    {
        if (x < 0 || x >= X) continue;
        int p = x + X * crosshair.y;
        temp = GRID_VALUE(dens,(x/upscaler),(crosshair.y/upscaler),slice);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;
        images->anatomy[slice].pixels[3*p] = (unsigned char) 255 * (alpha*red + (1-alpha)*temp);
        images->anatomy[slice].pixels[3*p+1] = (unsigned char) 255 * (alpha*green + (1-alpha)*temp);
        images->anatomy[slice].pixels[3*p+2] = (unsigned char) 255 * (alpha*blue + (1-alpha)*temp);
    }
    for (int y=crosshair.y-crossSize; y<crosshair.y+crossSize+1; y++)
    {
        if (y < 0 || y >= Y) continue;
        int p = crosshair.x + X * y;
        temp = GRID_VALUE(dens,(crosshair.x/upscaler),(y/upscaler),slice);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;
        images->anatomy[slice].pixels[3*p] = (unsigned char) 255 * (alpha*red + (1-alpha)*temp);
        images->anatomy[slice].pixels[3*p+1] = (unsigned char) 255 * (alpha*green + (1-alpha)*temp);
        images->anatomy[slice].pixels[3*p+2] = (unsigned char) 255 * (alpha*blue + (1-alpha)*temp);
    }

    return 1;
}
int add_error( FLOAT_GRID *dens, IMAGE_SET *images, float3 pos, float3 err, float red, float green, float blue, float alpha, int upscaler )
{
    float3 vec;
    vec.x = pos.x-err.x;
    vec.y = pos.y-err.y;
    vec.z = pos.z-err.z;
    float length = sqrt( vec.x*vec.x + vec.y*vec.y + vec.z*vec.z );
    float steps = (int)length * 4 * upscaler;

    for (int s=0; s<steps; s++)
    {
        float frac = (float)s/steps;
        int3 cross;
        cross.x = (int)floor(err.x + frac*vec.x + 0.5f);
        cross.y = (int)floor(err.y + frac*vec.y + 0.5f);
        cross.z = (int)floor(err.z + frac*vec.z + 0.5f);

        float temp = GRID_VALUE(dens,cross.x,cross.y,cross.z);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;

        int p = upscaler * cross.x + upscaler * dens->size.x * upscaler * cross.y;
        images->anatomy[cross.z].pixels[3*p] = (unsigned char) 255 * (alpha*red + (1-alpha)*temp);
        images->anatomy[cross.z].pixels[3*p+1] = (unsigned char) 255 * (alpha*green + (1-alpha)*temp);
        images->anatomy[cross.z].pixels[3*p+2] = (unsigned char) 255 * (alpha*blue + (1-alpha)*temp);
    }
    return 1;
}
int remove_crosshairs( FLOAT_GRID *dens, IMAGE_SET *images, float3 dpos, float red, float green, float blue, float alpha, int upscaler, int crossSize )
{
    int2 crosshair;
    crosshair.x = upscaler*(int)floor(dpos.x + 0.5f);
    crosshair.y = upscaler*(int)floor(dpos.y + 0.5f);

    int X = upscaler*dens->size.x;
    int Y = upscaler*dens->size.y;
    int slice = (int)floor(dpos.z + 0.5f);
    //printf("\n (%d,%d,%d)",crosshair.x,crosshair.y,slice);

    float temp;
    for (int x=crosshair.x-crossSize; x<crosshair.x+crossSize+1; x++)
    {
        if (x < 0 || x >= X) continue;
        int p = x + X * crosshair.y;
        temp = GRID_VALUE(dens,(x/upscaler),(crosshair.y/upscaler),slice);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;
        images->anatomy[slice].pixels[3*p] = (unsigned char) 255 * temp;
        images->anatomy[slice].pixels[3*p+1] = (unsigned char) 255 * temp;
        images->anatomy[slice].pixels[3*p+2] = (unsigned char) 255 * temp;
    }
    for (int y=crosshair.y-crossSize; y<crosshair.y+crossSize+1; y++)
    {
        if (y < 0 || y >= Y) continue;
        int p = crosshair.x + y * X;
        temp = GRID_VALUE(dens,(crosshair.x/upscaler),(y/upscaler),slice);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;
        images->anatomy[slice].pixels[3*p] = (unsigned char) 255 * temp;
        images->anatomy[slice].pixels[3*p+1] = (unsigned char) 255 * temp;
        images->anatomy[slice].pixels[3*p+2] = (unsigned char) 255 * temp;
    }

    return 1;
}
int remove_error( FLOAT_GRID *dens, IMAGE_SET *images, float3 pos, float3 err, float red, float green, float blue, float alpha, int upscaler )
{
    float3 vec;
    vec.x = pos.x-err.x;
    vec.y = pos.y-err.y;
    vec.z = pos.z-err.z;
    float length = sqrt( vec.x*vec.x + vec.y*vec.y + vec.z*vec.z );
    float steps = (int)length * 4 * upscaler;

    for (int s=0; s<steps; s++)
    {
        float frac = (float)s/steps;
        int3 cross;
        cross.x = (int)floor(err.x + frac*vec.x + 0.5f);
        cross.y = (int)floor(err.y + frac*vec.y + 0.5f);
        cross.z = (int)floor(err.z + frac*vec.z + 0.5f);

        float temp = GRID_VALUE(dens,cross.x,cross.y,cross.z);
        temp -= dens->min;
        temp /= (dens->max - dens->min);
        if (temp < 0) temp = 0;
        if (temp > 1) temp = 1;

        int p = upscaler * cross.x + upscaler * dens->size.x * upscaler * cross.y;
        images->anatomy[cross.z].pixels[3*p] = (unsigned char) 255 * temp;
        images->anatomy[cross.z].pixels[3*p+1] = (unsigned char) 255 * temp;
        images->anatomy[cross.z].pixels[3*p+2] = (unsigned char) 255 * temp;
    }
    return 1;
}







