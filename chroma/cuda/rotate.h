#ifndef __ROTATE_H__
#define __ROTATE_H__

#include "linalg.h"
#include "matrix.h"

__device__ const Matrix IDENTITY_MATRIX = {1,0,0,0,1,0,0,0,1};

__device__ Matrix
make_rotation_matrix(float phi, const float3 &n)
{
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    return IDENTITY_MATRIX*cos_phi + (1-cos_phi)*outer(n,n) +
	sin_phi*make_matrix(0,n.z,-n.y,-n.z,0,n.x,n.y,-n.x,0);
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
__device__ float3
rotate(const float3 &a, float phi, const float3 &n)
{
    float cos_phi = cosf(phi);
    float sin_phi = sinf(phi);

    return a*cos_phi + n*dot(a,n)*(1.0f-cos_phi) + cross(a,n)*sin_phi;
}

/* rotate points counterclockwise, when looking towards +infinity,
   through an angle `phi` about the axis `n`. */
__device__ float3
rotate_with_matrix(const float3 &a, float phi, const float3 &n)
{
    return make_rotation_matrix(phi,n)*a;
}

/**
 * Rotates a vector such that  under similar rotation the z vector [0,0,1] would align with a specified target direction.
 *
 * This method takes an input vector (original_vector) and rotates it so that
 * its z-axis aligns with the given target direction (target_direction).
 * The magnitude of the original vector is preserved.
 *
 * @param original_vector The input vector to be rotated.
 * @param target_direction The direction to which the z axis vector should be aligned.
 * @return A new vector aligned with the target direction.
 *  Uses Rodrigues' rotation formula:
    v' = v * cos(theta) + (k x v) * sin(theta) + k * (k · v) * (1 - cos(theta))

    ASSUMES TARGET VECTOR IS NORMALIZED
 */

__device__ float3 
rotate_zbasis_to_target_basis(float3 incident_vector, float3 target_vector){

    //z-axis unit vector
    float3 z = make_float3(0, 0, 1);
    
    //rotation axis is cross product of z and target_vector
    float3 k = cross(z, target_vector);
    
    //handle edge case where z and target_vector are aligned (no rotation needed)
    float k_norm = norm(k);
    if (k_norm < 1e-6f){
        return incident_vector;
    }
    //normalize the rotation axis
    k /= k_norm;
    
    //compute cos(theta) and sin(theta)
    float cos_theta = dot(z, target_vector);
    float sin_theta = k_norm;
    
    float3 output_vector = (incident_vector * cos_theta +
                    cross(k, incident_vector) * sin_theta +
                     k * dot(k, incident_vector) * (1 - cos_theta));
    
    return output_vector ;
}
#endif
