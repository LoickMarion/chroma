#ifndef __GEOMETRY_H__
#define __GEOMETRY_H__

#include "geometry_types.h"
#include "linalg.h"
#include "physical_constants.h"

__device__ float3 
to_float3(const uint3 &a)
{
  return make_float3(a.x, a.y, a.z);
}

__device__ uint4
get_packed_node(Geometry *geometry, const unsigned int &i)
{
    if (i < geometry->nprimary_nodes)
	return geometry->primary_nodes[i];
    else
	return geometry->extra_nodes[i - geometry->nprimary_nodes];
}
__device__ void
put_packed_node(Geometry *geometry, const unsigned int &i, const uint4 &node)
{
    if (i < geometry->nprimary_nodes)
	geometry->primary_nodes[i] = node;
    else
        geometry->extra_nodes[i - geometry->nprimary_nodes] = node;
}

__device__ Node
get_node(Geometry *geometry, const unsigned int &i)
{
    uint4 node = get_packed_node(geometry, i); 
	
    Node node_struct;

    uint3 lower_int = make_uint3(node.x & 0xFFFF, node.y & 0xFFFF, node.z & 0xFFFF);
    uint3 upper_int = make_uint3(node.x >> 16, node.y >> 16, node.z >> 16);


    node_struct.lower = geometry->world_origin + to_float3(lower_int) * geometry->world_scale;
    node_struct.upper = geometry->world_origin + to_float3(upper_int) * geometry->world_scale;
    node_struct.child = node.w & ~NCHILD_MASK;
    node_struct.nchild = node.w >> CHILD_BITS;
    
    return node_struct;
}

__device__ Triangle
get_triangle(Geometry *geometry, const unsigned int &i)
{
    uint3 triangle_data = geometry->triangles[i];

    Triangle triangle;
    triangle.v0 = geometry->vertices[triangle_data.x];
    triangle.v1 = geometry->vertices[triangle_data.y];
    triangle.v2 = geometry->vertices[triangle_data.z];

    return triangle;
}

template <class T>
__device__ float
interp_property(T *m, const float &x, const float *fp)
{
    if (x < m->wavelength_start)
	return fp[0];

    if (x > (m->wavelength_start + (m->wavelength_n-1)*m->wavelength_step))
	return fp[m->wavelength_n-1];

    int jl = (x-m->wavelength_start)/m->wavelength_step;

    return fp[jl] + (x-(m->wavelength_start + jl*m->wavelength_step))*(fp[jl+1]-fp[jl])/m->wavelength_step;
}

template <class T>
__device__ float
bilinear_interp_property(T *m, const float &wavelength, const float &angle, const float *fp)
{
    float angle_step = PI/(2.0f * m->num_angles);
    int angle_index = angle / angle_step;

    // If less than starting wavelength, interpolate along angle using minimum wavelength
    if (wavelength < m->wavelength_start) {
        float angle_frac = (angle - angle_index * angle_step) / angle_step;
        return fp[m->wavelength_n * angle_index] + angle_frac * (fp[m->wavelength_n * (angle_index + 1)] - fp[m->wavelength_n * angle_index]);
    }

    // If wavelength is greater than ending wavelength, interpolate along angle using maximum wavelength
    else if (wavelength >= (m->wavelength_start + (m->wavelength_n-1)*m->wavelength_step)) {
        float angle_frac = (angle - angle_index * angle_step) / angle_step;
        return fp[m->wavelength_n * (angle_index + 1) - 1] + angle_frac * (fp[m->wavelength_n * (angle_index + 2) - 1] - fp[m->wavelength_n * (angle_index + 1) - 1]);
    }
    
    // For intermediate values, perform bilinear interpolation
    else {
        int wavelength_index = (wavelength - m->wavelength_start) / m->wavelength_step;
        float wavelength_frac = (wavelength - (m->wavelength_start + wavelength_index * m->wavelength_step)) / m->wavelength_step;
        float angle_frac = (angle - angle_index * angle_step) / angle_step;

        // Find the 4 grid points for interpolation
        float lower_angle_lower_wl = fp[m->wavelength_n * angle_index + wavelength_index];
        float lower_angle_upper_wl = fp[m->wavelength_n * angle_index + wavelength_index + 1];
        float upper_angle_lower_wl = fp[m->wavelength_n * (angle_index + 1) + wavelength_index];
        float upper_angle_upper_wl = fp[m->wavelength_n * (angle_index + 1) + wavelength_index + 1];

        // Interpolate in the wavelength direction for both angles
        float lower_angle_wl_interp = lower_angle_lower_wl + wavelength_frac * (lower_angle_upper_wl - lower_angle_lower_wl);
        float upper_angle_wl_interp = upper_angle_lower_wl + wavelength_frac * (upper_angle_upper_wl - upper_angle_lower_wl);

        // Interpolate between the two angles
        return lower_angle_wl_interp + angle_frac * (upper_angle_wl_interp - lower_angle_wl_interp);
    }
}





#endif