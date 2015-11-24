#include <stdint.h>
#include "interlayer.h"
#include "float3.h"
#include "stencil.h"

// See interlayer.go for more details.
extern "C" __global__ void
addiecexchange(float* __restrict__ Bx, float* __restrict__ By, float* __restrict__ Bz,
            float* __restrict__ mx, float* __restrict__ my, float* __restrict__ mz,
            float* __restrict__ j1LUT2d, float* __restrict__ j2LUT2d, uint8_t* __restrict__ regions,
            float wx, float wy, float wz, int Nx, int Ny, int Nz, uint8_t PBC) {

	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	int iz = blockIdx.z * blockDim.z + threadIdx.z;

	if (ix >= Nx || iy >= Ny || iz >= Nz) {
		return;
	}

	// central cell
	int I = idx(ix, iy, iz);
	float3 m0 = make_float3(mx[I], my[I], mz[I]);

	if (is0(m0)) {
		return;
	}

	uint8_t r0 = regions[I];
	float3 B  = make_float3(Bx[I], By[I], Bz[I]);

	int i_;    // neighbor index
	float3 m_; // neighbor mag
	float j1__; // inter-cell bilinear exchange coefficient
	float j2__; // inter-cell biquadratic exchange coefficient

	// left neighbor
	i_  = idx(lclampx(ix-1), iy, iz);           // clamps or wraps index according to PBC
	m_  = make_float3(mx[i_], my[i_], mz[i_]);  // load m
	j1__ = j1LUT2d[matidx(r0, regions[i_])];
	j2__ = j2LUT2d[matidx(r0, regions[i_])];
	B += wx * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;

	// right neighbor
	i_  = idx(hclampx(ix+1), iy, iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	j1__ = j1LUT2d[matidx(r0, regions[i_])];
	j2__ = j2LUT2d[matidx(r0, regions[i_])];
	B += wx * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;

	// back neighbor
	i_  = idx(ix, lclampy(iy-1), iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	j1__ = j1LUT2d[matidx(r0, regions[i_])];
	j2__ = j2LUT2d[matidx(r0, regions[i_])];
	B += wy * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;

	// front neighbor
	i_  = idx(ix, hclampy(iy+1), iz);
	m_  = make_float3(mx[i_], my[i_], mz[i_]);
	j1__ = j1LUT2d[matidx(r0, regions[i_])];
	j2__ = j2LUT2d[matidx(r0, regions[i_])];
	B += wy * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;

	// only take vertical derivative for 3D sim
	if (Nz != 1) {
		// bottom neighbor
		i_  = idx(ix, iy, lclampz(iz-1));
		m_  = make_float3(mx[i_], my[i_], mz[i_]);
		j1__ = j1LUT2d[matidx(r0, regions[i_])];
		j2__ = j2LUT2d[matidx(r0, regions[i_])];
		B += wz * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;

		// top neighbor
		i_  = idx(ix, iy, hclampz(iz+1));
		m_  = make_float3(mx[i_], my[i_], mz[i_]);
		j1__ = j1LUT2d[matidx(r0, regions[i_])];
		j2__ = j2LUT2d[matidx(r0, regions[i_])];
		B += wz * (j1__ + 2 * j2__ * dot(m_, m0)) * m_;
	}

	Bx[I] = B.x;
	By[I] = B.y;
	Bz[I] = B.z;
}

