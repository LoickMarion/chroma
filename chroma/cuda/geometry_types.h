#ifndef __GEOMETRY_TYPES_H__
#define __GEOMETRY_TYPES_H__

struct Material
{
    float *refractive_index;
    float *absorption_length;
    float *scattering_length;
    unsigned int n;
    float step;
    float wavelength0;
};

struct Surface
{
    float *detect;
    float *absorb;
    float *reflect_diffuse;
    float *reflect_specular;
    unsigned int n;
    float step;
    float wavelength0;
};

struct Triangle
{
    float3 v0, v1, v2;
};

enum { INTERNAL_NODE, LEAF_NODE, PADDING_NODE };

struct Node
{
    float3 lower;
    float3 upper;
    unsigned int child;
    unsigned int kind;
};

struct Geometry
{
    float3 *vertices;
    uint3 *triangles;
    unsigned int *material_codes;
    unsigned int *colors;
    uint4 *nodes;
    Material **materials;
    Surface **surfaces;
    float3 world_origin;
    float world_scale;
    unsigned int branch_degree;
};

#endif
