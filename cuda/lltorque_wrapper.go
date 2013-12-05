package cuda

/*
 THIS FILE IS AUTO-GENERATED BY CUDA2GO.
 EDITING IS FUTILE.
*/

import (
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

var lltorque_code cu.Function

type lltorque_args struct {
	arg_tx       unsafe.Pointer
	arg_ty       unsafe.Pointer
	arg_tz       unsafe.Pointer
	arg_mx       unsafe.Pointer
	arg_my       unsafe.Pointer
	arg_mz       unsafe.Pointer
	arg_hx       unsafe.Pointer
	arg_hy       unsafe.Pointer
	arg_hz       unsafe.Pointer
	arg_alphaLUT unsafe.Pointer
	arg_regions  unsafe.Pointer
	arg_N        int
	argptr       [12]unsafe.Pointer
}

// Wrapper for lltorque CUDA kernel, asynchronous.
func k_lltorque_async(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, alphaLUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config, str cu.Stream) {
	if synchronous { // debug
		Sync()
	}

	if lltorque_code == 0 {
		lltorque_code = fatbinLoad(lltorque_map, "lltorque")
	}

	var _a_ lltorque_args

	_a_.arg_tx = tx
	_a_.argptr[0] = unsafe.Pointer(&_a_.arg_tx)
	_a_.arg_ty = ty
	_a_.argptr[1] = unsafe.Pointer(&_a_.arg_ty)
	_a_.arg_tz = tz
	_a_.argptr[2] = unsafe.Pointer(&_a_.arg_tz)
	_a_.arg_mx = mx
	_a_.argptr[3] = unsafe.Pointer(&_a_.arg_mx)
	_a_.arg_my = my
	_a_.argptr[4] = unsafe.Pointer(&_a_.arg_my)
	_a_.arg_mz = mz
	_a_.argptr[5] = unsafe.Pointer(&_a_.arg_mz)
	_a_.arg_hx = hx
	_a_.argptr[6] = unsafe.Pointer(&_a_.arg_hx)
	_a_.arg_hy = hy
	_a_.argptr[7] = unsafe.Pointer(&_a_.arg_hy)
	_a_.arg_hz = hz
	_a_.argptr[8] = unsafe.Pointer(&_a_.arg_hz)
	_a_.arg_alphaLUT = alphaLUT
	_a_.argptr[9] = unsafe.Pointer(&_a_.arg_alphaLUT)
	_a_.arg_regions = regions
	_a_.argptr[10] = unsafe.Pointer(&_a_.arg_regions)
	_a_.arg_N = N
	_a_.argptr[11] = unsafe.Pointer(&_a_.arg_N)

	args := _a_.argptr[:]
	cu.LaunchKernel(lltorque_code, cfg.Grid.X, cfg.Grid.Y, cfg.Grid.Z, cfg.Block.X, cfg.Block.Y, cfg.Block.Z, 0, str, args)

	if synchronous { // debug
		Sync()
	}
}

// Wrapper for lltorque CUDA kernel, synchronized.
func k_lltorque_sync(tx unsafe.Pointer, ty unsafe.Pointer, tz unsafe.Pointer, mx unsafe.Pointer, my unsafe.Pointer, mz unsafe.Pointer, hx unsafe.Pointer, hy unsafe.Pointer, hz unsafe.Pointer, alphaLUT unsafe.Pointer, regions unsafe.Pointer, N int, cfg *config) {
	Sync()
	k_lltorque_async(tx, ty, tz, mx, my, mz, hx, hy, hz, alphaLUT, regions, N, cfg, stream0)
	Sync()
}

var lltorque_map = map[int]string{0: "",
	20: lltorque_ptx_20,
	30: lltorque_ptx_30,
	35: lltorque_ptx_35}

const (
	lltorque_ptx_20 = `
.version 3.2
.target sm_20
.address_size 64


.visible .entry lltorque(
	.param .u64 lltorque_param_0,
	.param .u64 lltorque_param_1,
	.param .u64 lltorque_param_2,
	.param .u64 lltorque_param_3,
	.param .u64 lltorque_param_4,
	.param .u64 lltorque_param_5,
	.param .u64 lltorque_param_6,
	.param .u64 lltorque_param_7,
	.param .u64 lltorque_param_8,
	.param .u64 lltorque_param_9,
	.param .u64 lltorque_param_10,
	.param .u32 lltorque_param_11
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<35>;
	.reg .s64 	%rd<38>;


	ld.param.u64 	%rd12, [lltorque_param_0];
	ld.param.u64 	%rd13, [lltorque_param_1];
	ld.param.u64 	%rd14, [lltorque_param_2];
	ld.param.u64 	%rd10, [lltorque_param_3];
	ld.param.u64 	%rd11, [lltorque_param_4];
	ld.param.u64 	%rd15, [lltorque_param_5];
	ld.param.u64 	%rd16, [lltorque_param_6];
	ld.param.u64 	%rd17, [lltorque_param_7];
	ld.param.u64 	%rd18, [lltorque_param_8];
	ld.param.u64 	%rd19, [lltorque_param_9];
	ld.param.u64 	%rd20, [lltorque_param_10];
	ld.param.u32 	%r2, [lltorque_param_11];
	cvta.to.global.u64 	%rd1, %rd14;
	cvta.to.global.u64 	%rd2, %rd13;
	cvta.to.global.u64 	%rd3, %rd12;
	cvta.to.global.u64 	%rd4, %rd19;
	cvta.to.global.u64 	%rd5, %rd20;
	cvta.to.global.u64 	%rd6, %rd18;
	cvta.to.global.u64 	%rd7, %rd17;
	cvta.to.global.u64 	%rd8, %rd16;
	cvta.to.global.u64 	%rd9, %rd15;
	.loc 1 11 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 12 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd21, %rd10;
	.loc 1 14 1
	cvt.s64.s32	%rd22, %r1;
	mul.wide.s32 	%rd23, %r1, 4;
	add.s64 	%rd24, %rd21, %rd23;
	cvta.to.global.u64 	%rd25, %rd11;
	.loc 1 14 1
	add.s64 	%rd26, %rd25, %rd23;
	add.s64 	%rd27, %rd9, %rd23;
	.loc 1 15 1
	add.s64 	%rd28, %rd8, %rd23;
	add.s64 	%rd29, %rd7, %rd23;
	add.s64 	%rd30, %rd6, %rd23;
	.loc 1 16 1
	add.s64 	%rd31, %rd5, %rd22;
	ld.global.u8 	%rd32, [%rd31];
	shl.b64 	%rd33, %rd32, 2;
	add.s64 	%rd34, %rd4, %rd33;
	.loc 1 15 1
	ld.global.f32 	%f1, [%rd30];
	.loc 1 14 1
	ld.global.f32 	%f2, [%rd26];
	.loc 1 18 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 15 1
	ld.global.f32 	%f4, [%rd29];
	.loc 1 14 1
	ld.global.f32 	%f5, [%rd27];
	.loc 1 18 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	.loc 1 15 1
	ld.global.f32 	%f8, [%rd28];
	.loc 1 18 1
	mul.f32 	%f9, %f5, %f8;
	.loc 1 14 1
	ld.global.f32 	%f10, [%rd24];
	.loc 1 18 1
	mul.f32 	%f11, %f10, %f1;
	sub.f32 	%f12, %f9, %f11;
	mul.f32 	%f13, %f10, %f4;
	mul.f32 	%f14, %f2, %f8;
	sub.f32 	%f15, %f13, %f14;
	.loc 1 16 1
	ld.global.f32 	%f16, [%rd34];
	.loc 1 19 1
	fma.rn.f32 	%f17, %f16, %f16, 0f3F800000;
	mov.f32 	%f18, 0fBF800000;
	.loc 2 3608 3
	div.rn.f32 	%f19, %f18, %f17;
	.loc 1 20 1
	mul.f32 	%f20, %f2, %f15;
	mul.f32 	%f21, %f5, %f12;
	sub.f32 	%f22, %f20, %f21;
	mul.f32 	%f23, %f5, %f7;
	mul.f32 	%f24, %f10, %f15;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f10, %f12;
	mul.f32 	%f27, %f2, %f7;
	sub.f32 	%f28, %f26, %f27;
	fma.rn.f32 	%f29, %f16, %f22, %f7;
	fma.rn.f32 	%f30, %f16, %f25, %f12;
	fma.rn.f32 	%f31, %f16, %f28, %f15;
	mul.f32 	%f32, %f19, %f29;
	mul.f32 	%f33, %f19, %f30;
	mul.f32 	%f34, %f19, %f31;
	.loc 1 22 1
	add.s64 	%rd35, %rd3, %rd23;
	st.global.f32 	[%rd35], %f32;
	.loc 1 23 1
	add.s64 	%rd36, %rd2, %rd23;
	st.global.f32 	[%rd36], %f33;
	.loc 1 24 1
	add.s64 	%rd37, %rd1, %rd23;
	st.global.f32 	[%rd37], %f34;

BB0_2:
	.loc 1 26 2
	ret;
}


`
	lltorque_ptx_30 = `
.version 3.2
.target sm_30
.address_size 64


.visible .entry lltorque(
	.param .u64 lltorque_param_0,
	.param .u64 lltorque_param_1,
	.param .u64 lltorque_param_2,
	.param .u64 lltorque_param_3,
	.param .u64 lltorque_param_4,
	.param .u64 lltorque_param_5,
	.param .u64 lltorque_param_6,
	.param .u64 lltorque_param_7,
	.param .u64 lltorque_param_8,
	.param .u64 lltorque_param_9,
	.param .u64 lltorque_param_10,
	.param .u32 lltorque_param_11
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<35>;
	.reg .s64 	%rd<38>;


	ld.param.u64 	%rd12, [lltorque_param_0];
	ld.param.u64 	%rd13, [lltorque_param_1];
	ld.param.u64 	%rd14, [lltorque_param_2];
	ld.param.u64 	%rd10, [lltorque_param_3];
	ld.param.u64 	%rd11, [lltorque_param_4];
	ld.param.u64 	%rd15, [lltorque_param_5];
	ld.param.u64 	%rd16, [lltorque_param_6];
	ld.param.u64 	%rd17, [lltorque_param_7];
	ld.param.u64 	%rd18, [lltorque_param_8];
	ld.param.u64 	%rd19, [lltorque_param_9];
	ld.param.u64 	%rd20, [lltorque_param_10];
	ld.param.u32 	%r2, [lltorque_param_11];
	cvta.to.global.u64 	%rd1, %rd14;
	cvta.to.global.u64 	%rd2, %rd13;
	cvta.to.global.u64 	%rd3, %rd12;
	cvta.to.global.u64 	%rd4, %rd19;
	cvta.to.global.u64 	%rd5, %rd20;
	cvta.to.global.u64 	%rd6, %rd18;
	cvta.to.global.u64 	%rd7, %rd17;
	cvta.to.global.u64 	%rd8, %rd16;
	cvta.to.global.u64 	%rd9, %rd15;
	.loc 1 11 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 12 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd21, %rd10;
	.loc 1 14 1
	cvt.s64.s32	%rd22, %r1;
	mul.wide.s32 	%rd23, %r1, 4;
	add.s64 	%rd24, %rd21, %rd23;
	cvta.to.global.u64 	%rd25, %rd11;
	.loc 1 14 1
	add.s64 	%rd26, %rd25, %rd23;
	add.s64 	%rd27, %rd9, %rd23;
	.loc 1 15 1
	add.s64 	%rd28, %rd8, %rd23;
	add.s64 	%rd29, %rd7, %rd23;
	add.s64 	%rd30, %rd6, %rd23;
	.loc 1 16 1
	add.s64 	%rd31, %rd5, %rd22;
	ld.global.u8 	%rd32, [%rd31];
	shl.b64 	%rd33, %rd32, 2;
	add.s64 	%rd34, %rd4, %rd33;
	.loc 1 15 1
	ld.global.f32 	%f1, [%rd30];
	.loc 1 14 1
	ld.global.f32 	%f2, [%rd26];
	.loc 1 18 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 15 1
	ld.global.f32 	%f4, [%rd29];
	.loc 1 14 1
	ld.global.f32 	%f5, [%rd27];
	.loc 1 18 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	.loc 1 15 1
	ld.global.f32 	%f8, [%rd28];
	.loc 1 18 1
	mul.f32 	%f9, %f5, %f8;
	.loc 1 14 1
	ld.global.f32 	%f10, [%rd24];
	.loc 1 18 1
	mul.f32 	%f11, %f10, %f1;
	sub.f32 	%f12, %f9, %f11;
	mul.f32 	%f13, %f10, %f4;
	mul.f32 	%f14, %f2, %f8;
	sub.f32 	%f15, %f13, %f14;
	.loc 1 16 1
	ld.global.f32 	%f16, [%rd34];
	.loc 1 19 1
	fma.rn.f32 	%f17, %f16, %f16, 0f3F800000;
	mov.f32 	%f18, 0fBF800000;
	.loc 2 3608 3
	div.rn.f32 	%f19, %f18, %f17;
	.loc 1 20 1
	mul.f32 	%f20, %f2, %f15;
	mul.f32 	%f21, %f5, %f12;
	sub.f32 	%f22, %f20, %f21;
	mul.f32 	%f23, %f5, %f7;
	mul.f32 	%f24, %f10, %f15;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f10, %f12;
	mul.f32 	%f27, %f2, %f7;
	sub.f32 	%f28, %f26, %f27;
	fma.rn.f32 	%f29, %f16, %f22, %f7;
	fma.rn.f32 	%f30, %f16, %f25, %f12;
	fma.rn.f32 	%f31, %f16, %f28, %f15;
	mul.f32 	%f32, %f19, %f29;
	mul.f32 	%f33, %f19, %f30;
	mul.f32 	%f34, %f19, %f31;
	.loc 1 22 1
	add.s64 	%rd35, %rd3, %rd23;
	st.global.f32 	[%rd35], %f32;
	.loc 1 23 1
	add.s64 	%rd36, %rd2, %rd23;
	st.global.f32 	[%rd36], %f33;
	.loc 1 24 1
	add.s64 	%rd37, %rd1, %rd23;
	st.global.f32 	[%rd37], %f34;

BB0_2:
	.loc 1 26 2
	ret;
}


`
	lltorque_ptx_35 = `
.version 3.2
.target sm_35
.address_size 64


.weak .func  (.param .b32 func_retval0) cudaMalloc(
	.param .b64 cudaMalloc_param_0,
	.param .b64 cudaMalloc_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 66 3
	ret;
}

.weak .func  (.param .b32 func_retval0) cudaFuncGetAttributes(
	.param .b64 cudaFuncGetAttributes_param_0,
	.param .b64 cudaFuncGetAttributes_param_1
)
{
	.reg .s32 	%r<2>;


	mov.u32 	%r1, 30;
	st.param.b32	[func_retval0+0], %r1;
	.loc 2 71 3
	ret;
}

.visible .entry lltorque(
	.param .u64 lltorque_param_0,
	.param .u64 lltorque_param_1,
	.param .u64 lltorque_param_2,
	.param .u64 lltorque_param_3,
	.param .u64 lltorque_param_4,
	.param .u64 lltorque_param_5,
	.param .u64 lltorque_param_6,
	.param .u64 lltorque_param_7,
	.param .u64 lltorque_param_8,
	.param .u64 lltorque_param_9,
	.param .u64 lltorque_param_10,
	.param .u32 lltorque_param_11
)
{
	.reg .pred 	%p<2>;
	.reg .s16 	%rs<2>;
	.reg .s32 	%r<9>;
	.reg .f32 	%f<35>;
	.reg .s64 	%rd<39>;


	ld.param.u64 	%rd12, [lltorque_param_0];
	ld.param.u64 	%rd13, [lltorque_param_1];
	ld.param.u64 	%rd14, [lltorque_param_2];
	ld.param.u64 	%rd10, [lltorque_param_3];
	ld.param.u64 	%rd11, [lltorque_param_4];
	ld.param.u64 	%rd15, [lltorque_param_5];
	ld.param.u64 	%rd16, [lltorque_param_6];
	ld.param.u64 	%rd17, [lltorque_param_7];
	ld.param.u64 	%rd18, [lltorque_param_8];
	ld.param.u64 	%rd19, [lltorque_param_9];
	ld.param.u64 	%rd20, [lltorque_param_10];
	ld.param.u32 	%r2, [lltorque_param_11];
	cvta.to.global.u64 	%rd1, %rd14;
	cvta.to.global.u64 	%rd2, %rd13;
	cvta.to.global.u64 	%rd3, %rd12;
	cvta.to.global.u64 	%rd4, %rd19;
	cvta.to.global.u64 	%rd5, %rd20;
	cvta.to.global.u64 	%rd6, %rd18;
	cvta.to.global.u64 	%rd7, %rd17;
	cvta.to.global.u64 	%rd8, %rd16;
	cvta.to.global.u64 	%rd9, %rd15;
	.loc 1 11 1
	mov.u32 	%r3, %nctaid.x;
	mov.u32 	%r4, %ctaid.y;
	mov.u32 	%r5, %ctaid.x;
	mad.lo.s32 	%r6, %r3, %r4, %r5;
	mov.u32 	%r7, %ntid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r1, %r6, %r7, %r8;
	.loc 1 12 1
	setp.ge.s32	%p1, %r1, %r2;
	@%p1 bra 	BB2_2;

	cvta.to.global.u64 	%rd21, %rd10;
	.loc 1 14 1
	cvt.s64.s32	%rd22, %r1;
	mul.wide.s32 	%rd23, %r1, 4;
	add.s64 	%rd24, %rd21, %rd23;
	cvta.to.global.u64 	%rd25, %rd11;
	.loc 1 14 1
	add.s64 	%rd26, %rd25, %rd23;
	add.s64 	%rd27, %rd9, %rd23;
	.loc 1 15 1
	add.s64 	%rd28, %rd8, %rd23;
	add.s64 	%rd29, %rd7, %rd23;
	add.s64 	%rd30, %rd6, %rd23;
	.loc 1 16 1
	add.s64 	%rd31, %rd5, %rd22;
	ld.global.nc.u8 	%rs1, [%rd31];
	cvt.u64.u16	%rd32, %rs1;
	and.b64  	%rd33, %rd32, 255;
	shl.b64 	%rd34, %rd33, 2;
	add.s64 	%rd35, %rd4, %rd34;
	.loc 1 15 1
	ld.global.nc.f32 	%f1, [%rd30];
	.loc 1 14 1
	ld.global.nc.f32 	%f2, [%rd26];
	.loc 1 18 1
	mul.f32 	%f3, %f2, %f1;
	.loc 1 15 1
	ld.global.nc.f32 	%f4, [%rd29];
	.loc 1 14 1
	ld.global.nc.f32 	%f5, [%rd27];
	.loc 1 18 1
	mul.f32 	%f6, %f5, %f4;
	sub.f32 	%f7, %f3, %f6;
	.loc 1 15 1
	ld.global.nc.f32 	%f8, [%rd28];
	.loc 1 18 1
	mul.f32 	%f9, %f5, %f8;
	.loc 1 14 1
	ld.global.nc.f32 	%f10, [%rd24];
	.loc 1 18 1
	mul.f32 	%f11, %f10, %f1;
	sub.f32 	%f12, %f9, %f11;
	mul.f32 	%f13, %f10, %f4;
	mul.f32 	%f14, %f2, %f8;
	sub.f32 	%f15, %f13, %f14;
	.loc 1 16 1
	ld.global.nc.f32 	%f16, [%rd35];
	.loc 1 19 1
	fma.rn.f32 	%f17, %f16, %f16, 0f3F800000;
	mov.f32 	%f18, 0fBF800000;
	.loc 3 3608 3
	div.rn.f32 	%f19, %f18, %f17;
	.loc 1 20 1
	mul.f32 	%f20, %f2, %f15;
	mul.f32 	%f21, %f5, %f12;
	sub.f32 	%f22, %f20, %f21;
	mul.f32 	%f23, %f5, %f7;
	mul.f32 	%f24, %f10, %f15;
	sub.f32 	%f25, %f23, %f24;
	mul.f32 	%f26, %f10, %f12;
	mul.f32 	%f27, %f2, %f7;
	sub.f32 	%f28, %f26, %f27;
	fma.rn.f32 	%f29, %f16, %f22, %f7;
	fma.rn.f32 	%f30, %f16, %f25, %f12;
	fma.rn.f32 	%f31, %f16, %f28, %f15;
	mul.f32 	%f32, %f19, %f29;
	mul.f32 	%f33, %f19, %f30;
	mul.f32 	%f34, %f19, %f31;
	.loc 1 22 1
	add.s64 	%rd36, %rd3, %rd23;
	st.global.f32 	[%rd36], %f32;
	.loc 1 23 1
	add.s64 	%rd37, %rd2, %rd23;
	st.global.f32 	[%rd37], %f33;
	.loc 1 24 1
	add.s64 	%rd38, %rd1, %rd23;
	st.global.f32 	[%rd38], %f34;

BB2_2:
	.loc 1 26 2
	ret;
}


`
)
