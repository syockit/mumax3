package engine

// Interlayer exchange interaction
// See also cuda/interlayerexchange.cu

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/cuda/cu"
	"github.com/mumax/3/data"
	"unsafe"
)

var (
	B_iec     vAdder       // exchange field (T) output handle
	iecEx     iecExchParam // inter-cell exchange in 1e18 * Aex / Msat
	E_iec     *GetScalar   // Interlayer exchange energy
	Edens_iec sAdder       // Exchange energy density
)

func init() {
	B_iec.init("B_iec", "T", "Interlayer exchange field", AddIECField)
	E_iec = NewGetScalar("E_iec", "J", "Interlayer exchange energy", GetIECEnergy)
	Edens_iec.init("Edens_iec", "J/m3", "Interlayer exchange energy density", makeEdensAdder(&B_iec, -0.5))
	registerEnergy(GetExchangeEnergy, Edens_exch.AddTo)
	DeclFunc("ext_IECExchange", SetInterExchange, "Set exchange coupling energy between two regions.")
}

// Adds the current exchange field to dst
func AddIECField(dst *data.Slice) {
	if !iecEx.isZero() {
		cuda.AddIECExchange(dst, M.Buffer(), iecEx.Gpu(0), iecEx.Gpu(1), regions.Gpu(), M.Mesh())
	}
}

// Returns the current exchange energy in Joules.
func GetIECEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_iec)
}

// Set the interlayer exchange energy between region 1 and 2.
// It will disable conventional exchange.
func SetInterExchange(region1, region2 int, J1, J2 float64) {
	if (J1 != 0.0) || (J2 != 0.0) {
		ScaleInterExchange(region1, region2, 0.0)
	} else {
		ScaleInterExchange(region1, region2, 1.0)
	}
	iecEx.J1[symmidx(region1, region2)] = float32(J1)
	iecEx.J2[symmidx(region1, region2)] = float32(J2)
	iecEx.invalidate()
}

// stores interregion exchange stiffness
type iecExchParam struct {
	J1             [NREGION * (NREGION + 1) / 2]float32 // user defined J1 in regions (i,j)
	J2             [NREGION * (NREGION + 1) / 2]float32 // user defined J2 in regions (i,j)
	lutJ1          [NREGION * NREGION]float32           // derived 1e9 * J1/Msat in regions (i,j)
	lutJ2          [NREGION * NREGION]float32           // derived 1e9 * J2/Msat in regions (i,j)
	gpuJ1          cuda.SymmLUT                         // gpu copy of lut, lazily transferred when needed
	gpuJ2          cuda.SymmLUT
	gpu_ok, cpu_ok bool // gpu cache up-to date with lut source
}

// to be called after Aex, Msat or scaling changed
func (p *iecExchParam) invalidate() {
	p.cpu_ok = false
	p.gpu_ok = false
}

// Get a GPU mirror of the look-up table.
// Copies to GPU first only if needed.
func (p *iecExchParam) Gpu(j12 int) cuda.SymmLUT {
	p.update()
	if !p.gpu_ok {
		p.upload()
	}
	if j12 == 0 {
		return p.gpuJ1
	} else {
		return p.gpuJ2
	}
}

func (p *iecExchParam) update() {
	if !p.cpu_ok {
		msat := Msat.cpuLUT()

		for i := 0; i < NREGION; i++ {
			for j := i; j < NREGION; j++ {
				I := symmidx(i, j)
				p.lutJ1[matidx(i, j)] = 1e9 * safediv(iecEx.J1[I], msat[0][i])
				p.lutJ2[matidx(i, j)] = 1e9 * safediv(iecEx.J2[I], msat[0][i])
				p.lutJ1[matidx(j, i)] = 1e9 * safediv(iecEx.J1[I], msat[0][j])
				p.lutJ2[matidx(j, i)] = 1e9 * safediv(iecEx.J2[I], msat[0][j])
			}
		}
		p.gpu_ok = false
		p.cpu_ok = true
	}
}

func (p *iecExchParam) upload() {
	// alloc if  needed
	if p.gpuJ1 == nil {
		p.gpuJ1 = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lutJ1)) * cu.SIZEOF_FLOAT32))
	}
	if p.gpuJ2 == nil {
		p.gpuJ2 = cuda.SymmLUT(cuda.MemAlloc(int64(len(p.lutJ2)) * cu.SIZEOF_FLOAT32))
	}
	cuda.MemCpyHtoD(unsafe.Pointer(p.gpuJ1), unsafe.Pointer(&p.lutJ1[0]), cu.SIZEOF_FLOAT32*int64(len(p.lutJ1)))
	cuda.MemCpyHtoD(unsafe.Pointer(p.gpuJ2), unsafe.Pointer(&p.lutJ2[0]), cu.SIZEOF_FLOAT32*int64(len(p.lutJ2)))
	p.gpu_ok = true
}

func (p *iecExchParam) isZero() bool {
	for i := 0; i < (NREGION*NREGION+1)/2; i++ {
		if (p.J1[i] != 0) || (p.J2[i] != 0) {
			return false
		}
	}
	return true
}

func matidx(i, j int) int {
	return j*NREGION + i
}
