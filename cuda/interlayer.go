package cuda

import (
	"unsafe"

	"github.com/mumax/3/data"
)

// Add exchange field to Beff.
// 	m: normalized magnetization
// 	B: effective field in Tesla
// 	Aex_red: Aex / (Msat * 1e18 m2)
// see exchange.cu
func AddIECExchange(B, m *data.Slice, J1, J2 SymmLUT, regions *Bytes, mesh *data.Mesh) {
	c := mesh.CellSize()
	wx := float32(1e-9 / c[X])
	wy := float32(1e-9 / c[Y])
	wz := float32(1e-9 / c[Z])
	N := mesh.Size()
	pbc := mesh.PBC_code()
	cfg := make3DConf(N)
	k_addiecexchange_async(B.DevPtr(X), B.DevPtr(Y), B.DevPtr(Z),
		m.DevPtr(X), m.DevPtr(Y), m.DevPtr(Z),
		unsafe.Pointer(J1), unsafe.Pointer(J2), regions.Ptr,
		wx, wy, wz, N[X], N[Y], N[Z], pbc, cfg)
}
