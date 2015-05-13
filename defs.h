#ifndef DEFS_H_INCLUDED
#define DEFS_H_INCLUDED
#endif

#include <cuda_runtime_api.h>

#define PI 3.1416
#define GPU 0

/////////////////////////////// Structure Definitions ////////////////////////////////////////////////
class FLIST
{
  public:
    char name[256];
};

class DATASET
{
  public:
    FLIST *slice;
    bool load;
};


class ILIST
{
  public:
    unsigned char *pixels;
};


class IMAGE_SET
{
  public:
    ILIST *overlay;
    ILIST *anatomy;
    unsigned char *render_buf;
    int type;
};


class HUCURVE
{
  public:
    int length;
    float *hunits;
    float *dens;
};


class CNTR_SPECS
{
  public:
    int ROInumber;
    int3 rgb;
    char ROIname[100];

    int *CTRpoints;
    int TOTpoints;
    int subCntrs;
    float *matrix;

    bool draw;
};


class STRUCT_SET
{
  public:
    int CTRnumber;
    CNTR_SPECS object[100];
};


class SHIFT_SET
{
  public:
    char date[8];
    float4 shift;
};


class FLOAT_GRID
{
  public:
    float3 offset;
    float3 voxel;
    float3 startPos;
    float3 endPos;
    float4 shift;
    int3   size;
    float max;
    float min;
    float window;
    float level;
    float2 bracket;
    float *matrix;
};

class CHAR_GRID
{
  public:
    float3 start;
    float3 voxel;
    int3   size;
    float  max;
    float  min;
    int    slice;
    unsigned char *matrix;
};

class CTR_DOSES
{
  public:
    int     roi;
    float   vox;
    float   start_vol;
    float   end_vol;
    float   dvol;
    float3   avg;
    float3   max;
    float3   min;
    float4  *dvh_data;
    float   eud_coeff;
    int3    *voxels;
    float3 EUD;
};

class CTR_STATS
{
  public:
    int     roi;
    float   vox;
    float   max;
    float   min;
    float   avg;
    float   std;
    float   sum;
    float   pct;
};

class CTR_SET
{
  public:
    CTR_STATS jacobs[100];
    CTR_STATS gammas[100];
    CTR_DOSES doses[100];
    float4 com[100];
    float4 centroid[100];
    unsigned char *dvh_total;
    bool draw_dvh[100];
};

class STATS
{
  public:
    CTR_SET *week;
};

#define GRID_VALUE(GRID_ptr, i, j, k)\
    ((GRID_ptr)->matrix[(i) + (GRID_ptr)->size.x * ((j) + ((k) * (GRID_ptr)->size.y))])






