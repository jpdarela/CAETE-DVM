	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"funcs.f90"
	.def	PHOTO.;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO.
	.p2align	4, 0x90
PHOTO.:
	retq

	.def	PHOTO_mp_LEAP;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_mp_LEAP
	.p2align	4, 0x90
PHOTO_mp_LEAP:
	movl	(%rcx), %eax
	testb	$3, %al
	sete	%cl
	imull	$-1030792151, %eax, %eax
	addl	$85899344, %eax
	movl	%eax, %edx
	rorl	$2, %edx
	cmpl	$42949673, %edx
	setae	%dl
	rorl	$4, %eax
	cmpl	$10737419, %eax
	setb	%al
	orb	%dl, %al
	andb	%cl, %al
	movzbl	%al, %eax
	negl	%eax
	retq

	.def	PHOTO_mp_GROSS_PH;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@408f400000000000
	.section	.rdata,"dr",discard,__real@408f400000000000
	.p2align	3, 0x0
__real@408f400000000000:
	.quad	0x408f400000000000
	.globl	__real@bfe0000000000000
	.section	.rdata,"dr",discard,__real@bfe0000000000000
	.p2align	3, 0x0
__real@bfe0000000000000:
	.quad	0xbfe0000000000000
	.globl	__real@4000000000000000
	.section	.rdata,"dr",discard,__real@4000000000000000
	.p2align	3, 0x0
__real@4000000000000000:
	.quad	0x4000000000000000
	.globl	__real@bff8000000000000
	.section	.rdata,"dr",discard,__real@bff8000000000000
	.p2align	3, 0x0
__real@bff8000000000000:
	.quad	0xbff8000000000000
	.globl	__real@3fe5555555555555
	.section	.rdata,"dr",discard,__real@3fe5555555555555
	.p2align	3, 0x0
__real@3fe5555555555555:
	.quad	0x3fe5555555555555
	.globl	__real@41171d0ccccccccd
	.section	.rdata,"dr",discard,__real@41171d0ccccccccd
	.p2align	3, 0x0
__real@41171d0ccccccccd:
	.quad	0x41171d0ccccccccd
	.text
	.globl	PHOTO_mp_GROSS_PH
	.p2align	4, 0x90
PHOTO_mp_GROSS_PH:
.seh_proc PHOTO_mp_GROSS_PH
	subq	$120, %rsp
	.seh_stackalloc 120
	movaps	%xmm10, 96(%rsp)
	.seh_savexmm %xmm10, 96
	movaps	%xmm9, 80(%rsp)
	.seh_savexmm %xmm9, 80
	movaps	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movapd	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movsd	(%rdx), %xmm0
	mulsd	__real@408f400000000000(%rip), %xmm0
	movsd	(%rcx), %xmm8
	mulsd	(%r8), %xmm0
	xorpd	%xmm6, %xmm6
	maxsd	%xmm0, %xmm6
	movsd	__real@bfe0000000000000(%rip), %xmm7
	movapd	%xmm6, %xmm0
	mulsd	%xmm7, %xmm0
	callq	exp
	addsd	%xmm0, %xmm0
	movsd	__real@4000000000000000(%rip), %xmm9
	movapd	%xmm9, %xmm10
	subsd	%xmm0, %xmm10
	mulsd	%xmm10, %xmm7
	movapd	%xmm7, %xmm0
	callq	exp
	addsd	%xmm0, %xmm0
	subsd	%xmm0, %xmm9
	subsd	%xmm10, %xmm6
	mulsd	__real@bff8000000000000(%rip), %xmm6
	movapd	%xmm6, %xmm0
	callq	exp
	movsd	__real@3fe5555555555555(%rip), %xmm1
	mulsd	%xmm1, %xmm0
	subsd	%xmm0, %xmm1
	mulsd	__real@41171d0ccccccccd(%rip), %xmm8
	mulsd	%xmm9, %xmm8
	mulsd	%xmm1, %xmm8
	xorps	%xmm1, %xmm1
	cvtsd2ss	%xmm8, %xmm1
	xorpd	%xmm0, %xmm0
	maxss	%xmm1, %xmm0
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	movaps	80(%rsp), %xmm9
	movaps	96(%rsp), %xmm10
	addq	$120, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_LEAF_AREA_INDEX;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_mp_LEAF_AREA_INDEX
	.p2align	4, 0x90
PHOTO_mp_LEAF_AREA_INDEX:
	movsd	(%rcx), %xmm1
	mulsd	__real@408f400000000000(%rip), %xmm1
	mulsd	(%rdx), %xmm1
	xorpd	%xmm0, %xmm0
	maxsd	%xmm1, %xmm0
	retq

	.def	PHOTO_mp_SPEC_LEAF_AREA;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@4028000000000000
	.section	.rdata,"dr",discard,__real@4028000000000000
	.p2align	3, 0x0
__real@4028000000000000:
	.quad	0x4028000000000000
	.globl	__real@bfe19999a0000000
	.section	.rdata,"dr",discard,__real@bfe19999a0000000
	.p2align	3, 0x0
__real@bfe19999a0000000:
	.quad	0xbfe19999a0000000
	.globl	__real@3f9b3d07bcc00000
	.section	.rdata,"dr",discard,__real@3f9b3d07bcc00000
	.p2align	3, 0x0
__real@3f9b3d07bcc00000:
	.quad	0x3f9b3d07bcc00000
	.text
	.globl	PHOTO_mp_SPEC_LEAF_AREA
	.p2align	4, 0x90
PHOTO_mp_SPEC_LEAF_AREA:
.seh_proc PHOTO_mp_SPEC_LEAF_AREA
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movsd	(%rcx), %xmm0
	mulsd	__real@4028000000000000(%rip), %xmm0
	movsd	__real@bfe19999a0000000(%rip), %xmm1
	callq	pow
	mulsd	__real@3f9b3d07bcc00000(%rip), %xmm0
	addq	$40, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_SLA_REICH;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@4070a00000000000
	.section	.rdata,"dr",discard,__real@4070a00000000000
	.p2align	3, 0x0
__real@4070a00000000000:
	.quad	0x4070a00000000000
	.text
	.globl	PHOTO_mp_SLA_REICH
	.p2align	4, 0x90
PHOTO_mp_SLA_REICH:
.seh_proc PHOTO_mp_SLA_REICH
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movsd	(%rcx), %xmm0
	mulsd	__real@4028000000000000(%rip), %xmm0
	movsd	__real@bfe19999a0000000(%rip), %xmm1
	callq	pow
	mulsd	__real@4070a00000000000(%rip), %xmm0
	addq	$40, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_F_FOUR;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_mp_F_FOUR
	.p2align	4, 0x90
PHOTO_mp_F_FOUR:
.seh_proc PHOTO_mp_F_FOUR
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$64, %rsp
	.seh_stackalloc 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movapd	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%rcx, %rsi
	movsd	(%rdx), %xmm0
	mulsd	__real@408f400000000000(%rip), %xmm0
	mulsd	(%r8), %xmm0
	xorpd	%xmm6, %xmm6
	maxsd	%xmm0, %xmm6
	movsd	__real@bfe0000000000000(%rip), %xmm0
	mulsd	%xmm6, %xmm0
	callq	exp
	movapd	%xmm0, %xmm1
	addsd	%xmm0, %xmm1
	movsd	__real@4000000000000000(%rip), %xmm7
	movapd	%xmm7, %xmm0
	subsd	%xmm1, %xmm0
	subsd	%xmm0, %xmm6
	movl	(%rsi), %eax
	cmpl	$20, %eax
	je	.LBB6_1
	cmpl	$2, %eax
	je	.LBB6_5
	cmpl	$1, %eax
	jne	.LBB6_6
	mulsd	__real@bfe0000000000000(%rip), %xmm0
	callq	exp
	addsd	%xmm0, %xmm0
	subsd	%xmm0, %xmm7
	movapd	%xmm7, %xmm0
	jmp	.LBB6_6
.LBB6_5:
	mulsd	__real@bff8000000000000(%rip), %xmm6
	movapd	%xmm6, %xmm0
	callq	exp
	movapd	%xmm0, %xmm1
	movsd	__real@3fe5555555555555(%rip), %xmm0
	mulsd	%xmm0, %xmm1
	subsd	%xmm1, %xmm0
	jmp	.LBB6_6
.LBB6_1:
	movapd	%xmm6, %xmm0
.LBB6_6:
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	addq	$64, %rsp
	popq	%rsi
	retq
	.seh_endproc

	.def	PHOTO_mp_WATER_STRESS_MODIFIER;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@459c4000
	.section	.rdata,"dr",discard,__real@459c4000
	.p2align	2, 0x0
__real@459c4000:
	.long	0x459c4000
	.globl	__real@407f400000000000
	.section	.rdata,"dr",discard,__real@407f400000000000
	.p2align	3, 0x0
__real@407f400000000000:
	.quad	0x407f400000000000
	.globl	__real@412a5dfc21190ff3
	.section	.rdata,"dr",discard,__real@412a5dfc21190ff3
	.p2align	3, 0x0
__real@412a5dfc21190ff3:
	.quad	0x412a5dfc21190ff3
	.globl	__real@42c80000
	.section	.rdata,"dr",discard,__real@42c80000
	.p2align	2, 0x0
__real@42c80000:
	.long	0x42c80000
	.globl	__real@40b3880000000000
	.section	.rdata,"dr",discard,__real@40b3880000000000
	.p2align	3, 0x0
__real@40b3880000000000:
	.quad	0x40b3880000000000
	.globl	__real@4194996cf9db9476
	.section	.rdata,"dr",discard,__real@4194996cf9db9476
	.p2align	3, 0x0
__real@4194996cf9db9476:
	.quad	0x4194996cf9db9476
	.globl	__real@3ff64189374bc6a8
	.section	.rdata,"dr",discard,__real@3ff64189374bc6a8
	.p2align	3, 0x0
__real@3ff64189374bc6a8:
	.quad	0x3ff64189374bc6a8
	.globl	__real@4111310000000000
	.section	.rdata,"dr",discard,__real@4111310000000000
	.p2align	3, 0x0
__real@4111310000000000:
	.quad	0x4111310000000000
	.globl	__real@3ff0000000000000
	.section	.rdata,"dr",discard,__real@3ff0000000000000
	.p2align	3, 0x0
__real@3ff0000000000000:
	.quad	0x3ff0000000000000
	.globl	__real@bfb999999999999a
	.section	.rdata,"dr",discard,__real@bfb999999999999a
	.p2align	3, 0x0
__real@bfb999999999999a:
	.quad	0xbfb999999999999a
	.text
	.globl	PHOTO_mp_WATER_STRESS_MODIFIER
	.p2align	4, 0x90
PHOTO_mp_WATER_STRESS_MODIFIER:
.seh_proc PHOTO_mp_WATER_STRESS_MODIFIER
	subq	$56, %rsp
	.seh_stackalloc 56
	movapd	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	96(%rsp), %rax
	movsd	(%rcx), %xmm1
	divsd	(%rax), %xmm1
	movss	(%r8), %xmm3
	movss	(%r9), %xmm0
	cvtss2sd	%xmm0, %xmm2
	movsd	__real@407f400000000000(%rip), %xmm0
	ucomiss	__real@459c4000(%rip), %xmm3
	jbe	.LBB7_2
	mulsd	%xmm1, %xmm0
	mulsd	(%rdx), %xmm0
	movsd	__real@40b3880000000000(%rip), %xmm4
	jmp	.LBB7_4
.LBB7_2:
	mulsd	%xmm1, %xmm0
	mulsd	(%rdx), %xmm0
	ucomiss	__real@42c80000(%rip), %xmm3
	jbe	.LBB7_8
	cvtss2sd	%xmm3, %xmm4
.LBB7_4:
	movsd	__real@4194996cf9db9476(%rip), %xmm3
	divsd	%xmm4, %xmm3
	jmp	.LBB7_5
.LBB7_8:
	movsd	__real@412a5dfc21190ff3(%rip), %xmm3
.LBB7_5:
	mulsd	__real@3ff64189374bc6a8(%rip), %xmm2
	movsd	__real@4111310000000000(%rip), %xmm4
	divsd	%xmm3, %xmm4
	addsd	__real@3ff0000000000000(%rip), %xmm4
	divsd	%xmm4, %xmm2
	xorpd	%xmm6, %xmm6
	ucomisd	%xmm6, %xmm2
	jbe	.LBB7_7
	mulsd	__real@bfb999999999999a(%rip), %xmm0
	divsd	%xmm2, %xmm0
	callq	exp
	movsd	__real@3ff0000000000000(%rip), %xmm1
	subsd	%xmm0, %xmm1
.LBB7_7:
	maxsd	%xmm1, %xmm6
	movapd	%xmm6, %xmm0
	movaps	32(%rsp), %xmm6
	addq	$56, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_CANOPY_RESISTENCE;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3c23d70a
	.section	.rdata,"dr",discard,__real@3c23d70a
	.p2align	2, 0x0
__real@3c23d70a:
	.long	0x3c23d70a
	.globl	__real@40800000
	.section	.rdata,"dr",discard,__real@40800000
	.p2align	2, 0x0
__real@40800000:
	.long	0x40800000
	.globl	__real@c0400000
	.section	.rdata,"dr",discard,__real@c0400000
	.p2align	2, 0x0
__real@c0400000:
	.long	0xc0400000
	.globl	__real@bf000000
	.section	.rdata,"dr",discard,__real@bf000000
	.p2align	2, 0x0
__real@bf000000:
	.long	0xbf000000
	.globl	__real@3ff999999999999a
	.section	.rdata,"dr",discard,__real@3ff999999999999a
	.p2align	3, 0x0
__real@3ff999999999999a:
	.quad	0x3ff999999999999a
	.globl	__real@3f99ce075f6fd220
	.section	.rdata,"dr",discard,__real@3f99ce075f6fd220
	.p2align	3, 0x0
__real@3f99ce075f6fd220:
	.quad	0x3f99ce075f6fd220
	.globl	__real@3ec523a8a6a7ca08
	.section	.rdata,"dr",discard,__real@3ec523a8a6a7ca08
	.p2align	3, 0x0
__real@3ec523a8a6a7ca08:
	.quad	0x3ec523a8a6a7ca08
	.text
	.globl	PHOTO_mp_CANOPY_RESISTENCE
	.p2align	4, 0x90
PHOTO_mp_CANOPY_RESISTENCE:
	movss	(%rcx), %xmm0
	movss	__real@3c23d70a(%rip), %xmm1
	maxss	%xmm0, %xmm1
	movss	__real@40800000(%rip), %xmm2
	movaps	%xmm2, %xmm3
	cmpltss	%xmm0, %xmm3
	movaps	%xmm3, %xmm0
	andnps	%xmm1, %xmm0
	andps	%xmm2, %xmm3
	orps	%xmm0, %xmm3
	xorps	%xmm0, %xmm0
	rsqrtss	%xmm3, %xmm0
	mulss	%xmm0, %xmm3
	mulss	%xmm0, %xmm3
	addss	__real@c0400000(%rip), %xmm3
	mulss	__real@bf000000(%rip), %xmm0
	mulss	%xmm3, %xmm0
	cvtss2sd	%xmm0, %xmm0
	movsd	__real@3ff999999999999a(%rip), %xmm1
	movsd	(%r8), %xmm2
	mulsd	%xmm1, %xmm2
	mulsd	%xmm0, %xmm2
	movsd	(%rdx), %xmm0
	mulsd	__real@3f99ce075f6fd220(%rip), %xmm0
	addsd	%xmm1, %xmm2
	mulsd	%xmm2, %xmm0
	divsd	(%r9), %xmm0
	addsd	__real@3ec523a8a6a7ca08(%rip), %xmm0
	movsd	__real@3ff0000000000000(%rip), %xmm1
	divsd	%xmm0, %xmm1
	xorps	%xmm0, %xmm0
	cvtsd2ss	%xmm1, %xmm0
	movss	__real@459c4000(%rip), %xmm1
	minss	%xmm0, %xmm1
	movss	__real@42c80000(%rip), %xmm0
	maxss	%xmm1, %xmm0
	retq

	.def	PHOTO_mp_STOMATAL_CONDUCTANCE;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3ff99999a0000000
	.section	.rdata,"dr",discard,__real@3ff99999a0000000
	.p2align	3, 0x0
__real@3ff99999a0000000:
	.quad	0x3ff99999a0000000
	.text
	.globl	PHOTO_mp_STOMATAL_CONDUCTANCE
	.p2align	4, 0x90
PHOTO_mp_STOMATAL_CONDUCTANCE:
	movss	(%rcx), %xmm0
	xorps	%xmm1, %xmm1
	movaps	%xmm0, %xmm2
	cmpless	%xmm1, %xmm2
	movss	__real@3c23d70a(%rip), %xmm1
	andps	%xmm2, %xmm1
	andnps	%xmm0, %xmm2
	orps	%xmm1, %xmm2
	movss	__real@40800000(%rip), %xmm1
	movaps	%xmm1, %xmm3
	cmpltss	%xmm0, %xmm3
	movaps	%xmm3, %xmm0
	andnps	%xmm2, %xmm0
	andps	%xmm1, %xmm3
	orps	%xmm0, %xmm3
	xorps	%xmm0, %xmm0
	rsqrtss	%xmm3, %xmm0
	mulss	%xmm0, %xmm3
	mulss	%xmm0, %xmm3
	addss	__real@c0400000(%rip), %xmm3
	mulss	__real@bf000000(%rip), %xmm0
	mulss	%xmm3, %xmm0
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm0, %xmm1
	movsd	__real@3ff99999a0000000(%rip), %xmm2
	movsd	(%r8), %xmm0
	mulsd	%xmm2, %xmm0
	mulsd	%xmm1, %xmm0
	addsd	%xmm2, %xmm0
	mulsd	(%rdx), %xmm0
	divsd	(%r9), %xmm0
	retq

	.def	PHOTO_mp_WATER_UE;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@42237ae1
	.section	.rdata,"dr",discard,__real@42237ae1
	.p2align	2, 0x0
__real@42237ae1:
	.long	0x42237ae1
	.globl	__real@3dcccccd
	.section	.rdata,"dr",discard,__real@3dcccccd
	.p2align	2, 0x0
__real@3dcccccd:
	.long	0x3dcccccd
	.text
	.globl	PHOTO_mp_WATER_UE
	.p2align	4, 0x90
PHOTO_mp_WATER_UE:
	movss	(%rdx), %xmm1
	movss	(%r9), %xmm0
	mulss	__real@42237ae1(%rip), %xmm0
	mulss	__real@3dcccccd(%rip), %xmm1
	mulss	(%r8), %xmm1
	divss	%xmm1, %xmm0
	movsd	(%rcx), %xmm1
	xorpd	%xmm2, %xmm2
	ucomisd	%xmm1, %xmm2
	cvtsd2ss	%xmm1, %xmm1
	divss	%xmm0, %xmm1
	xorpd	%xmm2, %xmm2
	cmpeqss	%xmm2, %xmm0
	andnps	%xmm1, %xmm0
	jne	.LBB10_3
	jp	.LBB10_3
	xorps	%xmm0, %xmm0
.LBB10_3:
	retq

	.def	PHOTO_mp_TRANSPIRATION;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@42326666
	.section	.rdata,"dr",discard,__real@42326666
	.p2align	2, 0x0
__real@42326666:
	.long	0x42326666
	.globl	__real@3c9374bd
	.section	.rdata,"dr",discard,__real@3c9374bd
	.p2align	2, 0x0
__real@3c9374bd:
	.long	0x3c9374bd
	.text
	.globl	PHOTO_mp_TRANSPIRATION
	.p2align	4, 0x90
PHOTO_mp_TRANSPIRATION:
	movss	(%rcx), %xmm1
	movss	(%r8), %xmm0
	mulss	__real@42326666(%rip), %xmm0
	mulss	__real@3dcccccd(%rip), %xmm1
	mulss	(%rdx), %xmm1
	divss	%xmm1, %xmm0
	cmpl	$1, (%r9)
	je	.LBB11_2
	mulss	__real@3c9374bd(%rip), %xmm0
.LBB11_2:
	retq

	.def	PHOTO_mp_VAPOR_P_DEFICIT;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@41b849ba4195d4fe
	.section	.rdata,"dr",discard,__real@41b849ba4195d4fe
	.p2align	3, 0x0
__real@41b849ba4195d4fe:
	.long	0x4195d4fe
	.long	0x41b849ba
	.globl	__real@3b4464583b900901
	.section	.rdata,"dr",discard,__real@3b4464583b900901
	.p2align	3, 0x0
__real@3b4464583b900901:
	.long	0x3b900901
	.long	0x3b446458
	.globl	__real@438be8f64380ef5c
	.section	.rdata,"dr",discard,__real@438be8f64380ef5c
	.p2align	3, 0x0
__real@438be8f64380ef5c:
	.long	0x4380ef5c
	.long	0x438be8f6
	.globl	__real@40c3916840c39653
	.section	.rdata,"dr",discard,__real@40c3916840c39653
	.p2align	3, 0x0
__real@40c3916840c39653:
	.long	0x40c39653
	.long	0x40c39168
	.text
	.globl	PHOTO_mp_VAPOR_P_DEFICIT
	.p2align	4, 0x90
PHOTO_mp_VAPOR_P_DEFICIT:
.seh_proc PHOTO_mp_VAPOR_P_DEFICIT
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rdx, %rsi
	movss	(%rcx), %xmm1
	xorps	%xmm0, %xmm0
	xorl	%edi, %edi
	ucomiss	%xmm0, %xmm1
	setb	%dil
	leaq	__real@41b849ba4195d4fe(%rip), %rax
	movss	(%rax,%rdi,4), %xmm0
	leaq	__real@3b4464583b900901(%rip), %rax
	movss	(%rax,%rdi,4), %xmm2
	mulss	%xmm1, %xmm2
	subss	%xmm2, %xmm0
	leaq	__real@438be8f64380ef5c(%rip), %rax
	mulss	%xmm1, %xmm0
	addss	(%rax,%rdi,4), %xmm1
	divss	%xmm1, %xmm0
	callq	expf
	leaq	__real@40c3916840c39653(%rip), %rax
	mulss	(%rax,%rdi,4), %xmm0
	movss	(%rsi), %xmm1
	mulss	%xmm0, %xmm1
	subss	%xmm1, %xmm0
	mulss	__real@3dcccccd(%rip), %xmm0
	addq	$40, %rsp
	popq	%rdi
	popq	%rsi
	retq
	.seh_endproc

	.def	PHOTO_mp_REALIZED_NPP;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_mp_REALIZED_NPP
	.p2align	4, 0x90
PHOTO_mp_REALIZED_NPP:
	movsd	(%r8), %xmm2
	movsd	(%rdx), %xmm1
	xorl	%eax, %eax
	ucomisd	%xmm1, %xmm2
	sbbl	%eax, %eax
	ucomisd	%xmm1, %xmm2
	movq	40(%rsp), %rdx
	movsd	(%rcx), %xmm0
	jae	.LBB13_2
	mulsd	%xmm2, %xmm0
	divsd	%xmm1, %xmm0
	xorpd	%xmm1, %xmm1
	maxsd	%xmm1, %xmm0
.LBB13_2:
	movw	%ax, (%rdx)
	movsd	%xmm0, (%r9)
	retq

	.def	PHOTO_mp_VCMAX_A;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3fe6666666666666
	.section	.rdata,"dr",discard,__real@3fe6666666666666
	.p2align	3, 0x0
__real@3fe6666666666666:
	.quad	0x3fe6666666666666
	.globl	__real@bff28f5c28f5c28f
	.section	.rdata,"dr",discard,__real@bff28f5c28f5c28f
	.p2align	3, 0x0
__real@bff28f5c28f5c28f:
	.quad	0xbff28f5c28f5c28f
	.globl	__real@3feb333333333333
	.section	.rdata,"dr",discard,__real@3feb333333333333
	.p2align	3, 0x0
__real@3feb333333333333:
	.quad	0x3feb333333333333
	.globl	__real@bfd3333333333333
	.section	.rdata,"dr",discard,__real@bfd3333333333333
	.p2align	3, 0x0
__real@bfd3333333333333:
	.quad	0xbfd3333333333333
	.globl	__real@4024000000000000
	.section	.rdata,"dr",discard,__real@4024000000000000
	.p2align	3, 0x0
__real@4024000000000000:
	.quad	0x4024000000000000
	.globl	__real@3eb0c6f7a0b5ed8d
	.section	.rdata,"dr",discard,__real@3eb0c6f7a0b5ed8d
	.p2align	3, 0x0
__real@3eb0c6f7a0b5ed8d:
	.quad	0x3eb0c6f7a0b5ed8d
	.text
	.globl	PHOTO_mp_VCMAX_A
	.p2align	4, 0x90
PHOTO_mp_VCMAX_A:
.seh_proc PHOTO_mp_VCMAX_A
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$80, %rsp
	.seh_stackalloc 80
	movaps	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%r8, %rsi
	movsd	(%rcx), %xmm0
	movsd	(%rdx), %xmm6
	callq	log10
	movapd	%xmm0, %xmm7
	mulsd	__real@3fe6666666666666(%rip), %xmm7
	addsd	__real@bff28f5c28f5c28f(%rip), %xmm7
	movapd	%xmm6, %xmm0
	callq	log10
	movapd	%xmm0, %xmm6
	mulsd	__real@3feb333333333333(%rip), %xmm6
	addsd	__real@bfd3333333333333(%rip), %xmm6
	movsd	__real@4024000000000000(%rip), %xmm8
	movapd	%xmm8, %xmm0
	movapd	%xmm7, %xmm1
	callq	pow
	movapd	%xmm0, %xmm7
	movapd	%xmm8, %xmm0
	movapd	%xmm6, %xmm1
	callq	pow
	minsd	%xmm0, %xmm7
	mulsd	__real@3eb0c6f7a0b5ed8d(%rip), %xmm7
	divsd	(%rsi), %xmm7
	movapd	%xmm7, %xmm0
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	addq	$80, %rsp
	popq	%rsi
	retq
	.seh_endproc

	.def	PHOTO_mp_VCMAX_A1;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3fdb851eb851eb85
	.section	.rdata,"dr",discard,__real@3fdb851eb851eb85
	.p2align	3, 0x0
__real@3fdb851eb851eb85:
	.quad	0x3fdb851eb851eb85
	.globl	__real@bff8f5c28f5c28f6
	.section	.rdata,"dr",discard,__real@bff8f5c28f5c28f6
	.p2align	3, 0x0
__real@bff8f5c28f5c28f6:
	.quad	0xbff8f5c28f5c28f6
	.globl	__real@3fd7ae147ae147ae
	.section	.rdata,"dr",discard,__real@3fd7ae147ae147ae
	.p2align	3, 0x0
__real@3fd7ae147ae147ae:
	.quad	0x3fd7ae147ae147ae
	.globl	__real@3fdccccccccccccd
	.section	.rdata,"dr",discard,__real@3fdccccccccccccd
	.p2align	3, 0x0
__real@3fdccccccccccccd:
	.quad	0x3fdccccccccccccd
	.globl	__real@bfe999999999999a
	.section	.rdata,"dr",discard,__real@bfe999999999999a
	.p2align	3, 0x0
__real@bfe999999999999a:
	.quad	0xbfe999999999999a
	.globl	__real@3fd0000000000000
	.section	.rdata,"dr",discard,__real@3fd0000000000000
	.p2align	3, 0x0
__real@3fd0000000000000:
	.quad	0x3fd0000000000000
	.text
	.globl	PHOTO_mp_VCMAX_A1
	.p2align	4, 0x90
PHOTO_mp_VCMAX_A1:
.seh_proc PHOTO_mp_VCMAX_A1
	subq	$120, %rsp
	.seh_stackalloc 120
	movaps	%xmm10, 96(%rsp)
	.seh_savexmm %xmm10, 96
	movaps	%xmm9, 80(%rsp)
	.seh_savexmm %xmm9, 80
	movaps	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movsd	(%rcx), %xmm0
	movsd	(%rdx), %xmm7
	movsd	(%r8), %xmm6
	callq	log10
	movapd	%xmm0, %xmm8
	mulsd	__real@3fdb851eb851eb85(%rip), %xmm8
	addsd	__real@bff8f5c28f5c28f6(%rip), %xmm8
	movapd	%xmm6, %xmm0
	callq	log10
	movapd	%xmm0, %xmm9
	movsd	__real@3fd7ae147ae147ae(%rip), %xmm10
	mulsd	%xmm0, %xmm10
	addsd	%xmm8, %xmm10
	movapd	%xmm7, %xmm0
	callq	log10
	mulsd	__real@3fdccccccccccccd(%rip), %xmm0
	addsd	__real@bfe999999999999a(%rip), %xmm0
	mulsd	__real@3fd0000000000000(%rip), %xmm9
	addsd	%xmm0, %xmm9
	movsd	__real@4024000000000000(%rip), %xmm7
	movapd	%xmm7, %xmm0
	movapd	%xmm10, %xmm1
	callq	pow
	movapd	%xmm0, %xmm8
	movapd	%xmm7, %xmm0
	movapd	%xmm9, %xmm1
	callq	pow
	minsd	%xmm0, %xmm8
	divsd	%xmm6, %xmm8
	movapd	%xmm8, %xmm0
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	movaps	80(%rsp), %xmm9
	movaps	96(%rsp), %xmm10
	addq	$120, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_VCMAX_B;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3fe199999999999a
	.section	.rdata,"dr",discard,__real@3fe199999999999a
	.p2align	3, 0x0
__real@3fe199999999999a:
	.quad	0x3fe199999999999a
	.globl	__real@3ff91eb851eb851f
	.section	.rdata,"dr",discard,__real@3ff91eb851eb851f
	.p2align	3, 0x0
__real@3ff91eb851eb851f:
	.quad	0x3ff91eb851eb851f
	.text
	.globl	PHOTO_mp_VCMAX_B
	.p2align	4, 0x90
PHOTO_mp_VCMAX_B:
.seh_proc PHOTO_mp_VCMAX_B
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movsd	(%rcx), %xmm0
	callq	log10
	movapd	%xmm0, %xmm1
	mulsd	__real@3fe199999999999a(%rip), %xmm1
	addsd	__real@3ff91eb851eb851f(%rip), %xmm1
	movsd	__real@4024000000000000(%rip), %xmm0
	callq	pow
	mulsd	__real@3eb0c6f7a0b5ed8d(%rip), %xmm0
	addq	$40, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_PHOTOSYNTHESIS_RATE;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@bfd0000000000000
	.section	.rdata,"dr",discard,__real@bfd0000000000000
	.p2align	3, 0x0
__real@bfd0000000000000:
	.quad	0xbfd0000000000000
	.globl	__real@3fd99999a0000000
	.section	.rdata,"dr",discard,__real@3fd99999a0000000
	.p2align	3, 0x0
__real@3fd99999a0000000:
	.quad	0x3fd99999a0000000
	.globl	__real@3f14f8b588e368f1
	.section	.rdata,"dr",discard,__real@3f14f8b588e368f1
	.p2align	3, 0x0
__real@3f14f8b588e368f1:
	.quad	0x3f14f8b588e368f1
	.globl	__real@3fb999999999999a
	.section	.rdata,"dr",discard,__real@3fb999999999999a
	.p2align	3, 0x0
__real@3fb999999999999a:
	.quad	0x3fb999999999999a
	.globl	__real@c004000000000000
	.section	.rdata,"dr",discard,__real@c004000000000000
	.p2align	3, 0x0
__real@c004000000000000:
	.quad	0xc004000000000000
	.globl	__real@3fd3333333333333
	.section	.rdata,"dr",discard,__real@3fd3333333333333
	.p2align	3, 0x0
__real@3fd3333333333333:
	.quad	0x3fd3333333333333
	.globl	__real@c025999999999999
	.section	.rdata,"dr",discard,__real@c025999999999999
	.p2align	3, 0x0
__real@c025999999999999:
	.quad	0xc025999999999999
	.globl	__real@43889333
	.section	.rdata,"dr",discard,__real@43889333
	.p2align	2, 0x0
__real@43889333:
	.long	0x43889333
	.globl	__real@3fb99999a0000000
	.section	.rdata,"dr",discard,__real@3fb99999a0000000
	.p2align	3, 0x0
__real@3fb99999a0000000:
	.quad	0x3fb99999a0000000
	.globl	__real@c03dd0a3d440f5c0
	.section	.rdata,"dr",discard,__real@c03dd0a3d440f5c0
	.p2align	3, 0x0
__real@c03dd0a3d440f5c0:
	.quad	0xc03dd0a3d440f5c0
	.globl	__real@4000ccccc0000000
	.section	.rdata,"dr",discard,__real@4000ccccc0000000
	.p2align	3, 0x0
__real@4000ccccc0000000:
	.quad	0x4000ccccc0000000
	.globl	__real@4054800000000000
	.section	.rdata,"dr",discard,__real@4054800000000000
	.p2align	3, 0x0
__real@4054800000000000:
	.quad	0x4054800000000000
	.globl	__real@4082780000000000
	.section	.rdata,"dr",discard,__real@4082780000000000
	.p2align	3, 0x0
__real@4082780000000000:
	.quad	0x4082780000000000
	.globl	__real@c106979800000000
	.section	.rdata,"dr",discard,__real@c106979800000000
	.p2align	3, 0x0
__real@c106979800000000:
	.quad	0xc106979800000000
	.globl	__real@4020a0c49ba5e354
	.section	.rdata,"dr",discard,__real@4020a0c49ba5e354
	.p2align	3, 0x0
__real@4020a0c49ba5e354:
	.quad	0x4020a0c49ba5e354
	.globl	__real@c0bc77f2b2b6a7a1
	.section	.rdata,"dr",discard,__real@c0bc77f2b2b6a7a1
	.p2align	3, 0x0
__real@c0bc77f2b2b6a7a1:
	.quad	0xc0bc77f2b2b6a7a1
	.globl	__real@403871a34c9ff892
	.section	.rdata,"dr",discard,__real@403871a34c9ff892
	.p2align	3, 0x0
__real@403871a34c9ff892:
	.quad	0x403871a34c9ff892
	.globl	__real@4055d8000b607651
	.section	.rdata,"dr",discard,__real@4055d8000b607651
	.p2align	3, 0x0
__real@4055d8000b607651:
	.quad	0x4055d8000b607651
	.globl	__real@41ff0d8891000002
	.section	.rdata,"dr",discard,__real@41ff0d8891000002
	.p2align	3, 0x0
__real@41ff0d8891000002:
	.quad	0x41ff0d8891000002
	.globl	__real@bd7b0f6a
	.section	.rdata,"dr",discard,__real@bd7b0f6a
	.p2align	2, 0x0
__real@bd7b0f6a:
	.long	0xbd7b0f6a
	.globl	__real@3fd645a2
	.section	.rdata,"dr",discard,__real@3fd645a2
	.p2align	2, 0x0
__real@3fd645a2:
	.long	0x3fd645a2
	.globl	__real@3a99326c
	.section	.rdata,"dr",discard,__real@3a99326c
	.p2align	2, 0x0
__real@3a99326c:
	.long	0x3a99326c
	.globl	__real@b714e1f8
	.section	.rdata,"dr",discard,__real@b714e1f8
	.p2align	2, 0x0
__real@b714e1f8:
	.long	0xb714e1f8
	.globl	__real@3feeaf57959e18bb
	.section	.rdata,"dr",discard,__real@3feeaf57959e18bb
	.p2align	3, 0x0
__real@3feeaf57959e18bb:
	.quad	0x3feeaf57959e18bb
	.globl	__real@3fb75f6fd1228000
	.section	.rdata,"dr",discard,__real@3fb75f6fd1228000
	.p2align	3, 0x0
__real@3fb75f6fd1228000:
	.quad	0x3fb75f6fd1228000
	.globl	__real@3fe0000000000000
	.section	.rdata,"dr",discard,__real@3fe0000000000000
	.p2align	3, 0x0
__real@3fe0000000000000:
	.quad	0x3fe0000000000000
	.globl	__real@bffdc28f5c28f5c3
	.section	.rdata,"dr",discard,__real@bffdc28f5c28f5c3
	.p2align	3, 0x0
__real@bffdc28f5c28f5c3:
	.quad	0xbffdc28f5c28f5c3
	.globl	__xmm@80000000000000008000000000000000
	.section	.rdata,"dr",discard,__xmm@80000000000000008000000000000000
	.p2align	4, 0x0
__xmm@80000000000000008000000000000000:
	.quad	0x8000000000000000
	.quad	0x8000000000000000
	.globl	__real@3fe23d70a3d70a3d
	.section	.rdata,"dr",discard,__real@3fe23d70a3d70a3d
	.p2align	3, 0x0
__real@3fe23d70a3d70a3d:
	.quad	0x3fe23d70a3d70a3d
	.globl	__real@40104ec4ec4ec4ec
	.section	.rdata,"dr",discard,__real@40104ec4ec4ec4ec
	.p2align	3, 0x0
__real@40104ec4ec4ec4ec:
	.quad	0x40104ec4ec4ec4ec
	.globl	__real@4000cccccccccccd
	.section	.rdata,"dr",discard,__real@4000cccccccccccd
	.p2align	3, 0x0
__real@4000cccccccccccd:
	.quad	0x4000cccccccccccd
	.globl	__real@403e000000000000
	.section	.rdata,"dr",discard,__real@403e000000000000
	.p2align	3, 0x0
__real@403e000000000000:
	.quad	0x403e000000000000
	.globl	__real@3ffc36c3627f4dc0
	.section	.rdata,"dr",discard,__real@3ffc36c3627f4dc0
	.p2align	3, 0x0
__real@3ffc36c3627f4dc0:
	.quad	0x3ffc36c3627f4dc0
	.globl	__real@3feccccccccccccd
	.section	.rdata,"dr",discard,__real@3feccccccccccccd
	.p2align	3, 0x0
__real@3feccccccccccccd:
	.quad	0x3feccccccccccccd
	.globl	__real@3fb9db211e1a18ba
	.section	.rdata,"dr",discard,__real@3fb9db211e1a18ba
	.p2align	3, 0x0
__real@3fb9db211e1a18ba:
	.quad	0x3fb9db211e1a18ba
	.globl	__real@3ff3333333333333
	.section	.rdata,"dr",discard,__real@3ff3333333333333
	.p2align	3, 0x0
__real@3ff3333333333333:
	.quad	0x3ff3333333333333
	.globl	__real@3fe69d0369d0369d
	.section	.rdata,"dr",discard,__real@3fe69d0369d0369d
	.p2align	3, 0x0
__real@3fe69d0369d0369d:
	.quad	0x3fe69d0369d0369d
	.globl	__real@40204ec4ec4ec4ec
	.section	.rdata,"dr",discard,__real@40204ec4ec4ec4ec
	.p2align	3, 0x0
__real@40204ec4ec4ec4ec:
	.quad	0x40204ec4ec4ec4ec
	.globl	__real@3fb16872b020c49c
	.section	.rdata,"dr",discard,__real@3fb16872b020c49c
	.p2align	3, 0x0
__real@3fb16872b020c49c:
	.quad	0x3fb16872b020c49c
	.globl	__real@c00a8f5c28f5c28f
	.section	.rdata,"dr",discard,__real@c00a8f5c28f5c28f
	.p2align	3, 0x0
__real@c00a8f5c28f5c28f:
	.quad	0xc00a8f5c28f5c28f
	.globl	__real@3fe346f0940c565d
	.section	.rdata,"dr",discard,__real@3fe346f0940c565d
	.p2align	3, 0x0
__real@3fe346f0940c565d:
	.quad	0x3fe346f0940c565d
	.globl	__real@3fe1344d1344d134
	.section	.rdata,"dr",discard,__real@3fe1344d1344d134
	.p2align	3, 0x0
__real@3fe1344d1344d134:
	.quad	0x3fe1344d1344d134
	.text
	.globl	PHOTO_mp_PHOTOSYNTHESIS_RATE
	.p2align	4, 0x90
PHOTO_mp_PHOTOSYNTHESIS_RATE:
.seh_proc PHOTO_mp_PHOTOSYNTHESIS_RATE
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$168, %rsp
	.seh_stackalloc 168
	movaps	%xmm13, 144(%rsp)
	.seh_savexmm %xmm13, 144
	movaps	%xmm12, 128(%rsp)
	.seh_savexmm %xmm12, 128
	movaps	%xmm11, 112(%rsp)
	.seh_savexmm %xmm11, 112
	movaps	%xmm10, 96(%rsp)
	.seh_savexmm %xmm10, 96
	movaps	%xmm9, 80(%rsp)
	.seh_savexmm %xmm9, 80
	movaps	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%r9, %rdi
	movq	%r8, %rbx
	movq	%rdx, %r14
	movq	%rcx, %rsi
	movq	336(%rsp), %r12
	movq	272(%rsp), %r13
	movq	280(%rsp), %rbp
	movq	328(%rsp), %r15
	movq	304(%rsp), %rax
	movq	296(%rsp), %rcx
	movq	312(%rsp), %rdx
	movsd	(%rdx), %xmm11
	mulsd	__real@bfd0000000000000(%rip), %xmm11
	movq	288(%rsp), %rdx
	movsd	__real@3ff0000000000000(%rip), %xmm10
	addsd	%xmm10, %xmm11
	movsd	__real@3fd99999a0000000(%rip), %xmm7
	movsd	(%rdx), %xmm8
	mulsd	%xmm7, %xmm8
	mulsd	(%rcx), %xmm7
	minsd	%xmm10, %xmm11
	movsd	(%rax), %xmm0
	mulsd	__real@4028000000000000(%rip), %xmm0
	movsd	__real@bfe19999a0000000(%rip), %xmm1
	callq	pow
	movapd	%xmm0, %xmm6
	mulsd	__real@3f9b3d07bcc00000(%rip), %xmm6
	movapd	%xmm8, %xmm0
	callq	log10
	movapd	%xmm0, %xmm8
	movsd	__real@3fe6666666666666(%rip), %xmm12
	mulsd	%xmm12, %xmm8
	addsd	__real@bff28f5c28f5c28f(%rip), %xmm8
	movapd	%xmm7, %xmm0
	callq	log10
	movapd	%xmm0, %xmm7
	mulsd	__real@3feb333333333333(%rip), %xmm7
	addsd	__real@bfd3333333333333(%rip), %xmm7
	movsd	__real@4024000000000000(%rip), %xmm9
	movapd	%xmm9, %xmm0
	movapd	%xmm8, %xmm1
	callq	pow
	movapd	%xmm0, %xmm8
	movapd	%xmm9, %xmm0
	movapd	%xmm7, %xmm1
	callq	pow
	minsd	%xmm0, %xmm8
	mulsd	__real@3eb0c6f7a0b5ed8d(%rip), %xmm11
	mulsd	%xmm8, %xmm11
	divsd	%xmm6, %xmm11
	movsd	__real@3f14f8b588e368f1(%rip), %xmm9
	movapd	%xmm9, %xmm13
	minsd	%xmm11, %xmm13
	mulsd	%xmm12, %xmm13
	movsd	%xmm13, (%r15)
	movss	(%r14), %xmm11
	xorps	%xmm6, %xmm6
	cvtss2sd	%xmm11, %xmm6
	movsd	__real@3fb999999999999a(%rip), %xmm8
	mulsd	%xmm6, %xmm8
	addsd	__real@c004000000000000(%rip), %xmm8
	movapd	%xmm8, %xmm0
	callq	exp2
	movapd	%xmm0, %xmm7
	mulsd	%xmm13, %xmm7
	mulsd	__real@3fd3333333333333(%rip), %xmm6
	addsd	__real@c025999999999999(%rip), %xmm6
	movapd	%xmm6, %xmm0
	callq	exp
	addsd	%xmm10, %xmm0
	divsd	%xmm0, %xmm7
	minsd	%xmm7, %xmm9
	cmpl	$0, (%rbp)
	je	.LBB17_1
	movss	__real@43889333(%rip), %xmm0
	addss	%xmm11, %xmm0
	xorps	%xmm12, %xmm12
	cvtss2sd	%xmm0, %xmm12
	movsd	__real@3fb99999a0000000(%rip), %xmm1
	mulsd	%xmm12, %xmm1
	addsd	__real@c03dd0a3d440f5c0(%rip), %xmm1
	movsd	__real@4000ccccc0000000(%rip), %xmm0
	callq	pow
	movapd	%xmm0, %xmm6
	mulsd	__real@4054800000000000(%rip), %xmm6
	cmpl	$1, (%r13)
	movss	(%rdi), %xmm0
	xorps	%xmm8, %xmm8
	cvtss2sd	%xmm0, %xmm8
	je	.LBB17_6
	mulsd	__real@3fe6666666666666(%rip), %xmm8
.LBB17_6:
	movsd	__real@4082780000000000(%rip), %xmm0
	mulsd	%xmm12, %xmm0
	addsd	__real@c106979800000000(%rip), %xmm0
	movsd	__real@4020a0c49ba5e354(%rip), %xmm1
	mulsd	%xmm12, %xmm1
	divsd	%xmm1, %xmm0
	callq	exp
	movapd	%xmm0, %xmm7
	addsd	%xmm10, %xmm7
	movsd	__real@c0bc77f2b2b6a7a1(%rip), %xmm0
	divsd	%xmm12, %xmm0
	addsd	__real@403871a34c9ff892(%rip), %xmm0
	callq	exp
	mulsd	__real@4055d8000b607651(%rip), %xmm0
	divsd	%xmm7, %xmm0
	movapd	%xmm8, %xmm1
	mulsd	%xmm8, %xmm1
	mulsd	__real@41ff0d8891000002(%rip), %xmm1
	mulsd	%xmm0, %xmm0
	divsd	%xmm0, %xmm1
	addsd	%xmm10, %xmm1
	xorps	%xmm0, %xmm0
	sqrtsd	%xmm1, %xmm0
	movss	__real@bd7b0f6a(%rip), %xmm1
	mulss	%xmm11, %xmm1
	addss	__real@3fd645a2(%rip), %xmm1
	cvtss2sd	%xmm1, %xmm1
	movaps	%xmm11, %xmm2
	mulss	%xmm11, %xmm2
	mulss	%xmm2, %xmm11
	mulss	__real@3a99326c(%rip), %xmm2
	cvtss2sd	%xmm2, %xmm2
	addsd	%xmm1, %xmm2
	mulss	__real@b714e1f8(%rip), %xmm11
	xorps	%xmm1, %xmm1
	cvtss2sd	%xmm11, %xmm1
	addsd	%xmm2, %xmm1
	mulsd	__real@3feeaf57959e18bb(%rip), %xmm1
	mulsd	(%rsi), %xmm1
	addsd	%xmm1, %xmm6
	mulsd	%xmm0, %xmm6
	mulsd	__real@3fb75f6fd1228000(%rip), %xmm8
	mulsd	%xmm1, %xmm8
	divsd	%xmm6, %xmm8
	movsd	%xmm8, (%r12)
	movsd	__real@3fe0000000000000(%rip), %xmm0
	mulsd	%xmm9, %xmm0
	addsd	%xmm8, %xmm0
	movapd	%xmm0, %xmm1
	mulsd	%xmm0, %xmm1
	mulsd	__real@bffdc28f5c28f5c3(%rip), %xmm9
	mulsd	%xmm8, %xmm9
	jmp	.LBB17_7
.LBB17_1:
	movapd	__xmm@80000000000000008000000000000000(%rip), %xmm7
	xorpd	%xmm8, %xmm7
	movsd	__real@3fe23d70a3d70a3d(%rip), %xmm0
	movapd	%xmm7, %xmm1
	callq	pow
	movapd	%xmm0, %xmm6
	movsd	__real@40104ec4ec4ec4ec(%rip), %xmm10
	mulsd	%xmm0, %xmm10
	movsd	__real@4000cccccccccccd(%rip), %xmm0
	movapd	%xmm8, %xmm1
	callq	pow
	movapd	%xmm0, %xmm8
	mulsd	__real@403e000000000000(%rip), %xmm8
	xorpd	%xmm0, %xmm0
	xorl	%r14d, %r14d
	ucomiss	%xmm0, %xmm11
	setb	%r14b
	leaq	__real@41b849ba4195d4fe(%rip), %rax
	movss	(%rax,%r14,4), %xmm0
	leaq	__real@3b4464583b900901(%rip), %rax
	movss	(%rax,%r14,4), %xmm1
	mulss	%xmm11, %xmm1
	subss	%xmm1, %xmm0
	leaq	__real@438be8f64380ef5c(%rip), %rax
	mulss	%xmm11, %xmm0
	addss	(%rax,%r14,4), %xmm11
	divss	%xmm11, %xmm0
	callq	expf
	leaq	__real@40c3916840c39653(%rip), %rax
	mulss	(%rax,%r14,4), %xmm0
	cvtss2sd	%xmm0, %xmm0
	movss	(%rbx), %xmm1
	cvtss2sd	%xmm1, %xmm1
	subsd	%xmm0, %xmm1
	mulsd	__real@3ffc36c3627f4dc0(%rip), %xmm0
	divsd	%xmm1, %xmm0
	addsd	__real@3feccccccccccccd(%rip), %xmm0
	movsd	(%rsi), %xmm11
	mulsd	__real@3fb9db211e1a18ba(%rip), %xmm11
	subsd	%xmm10, %xmm11
	mulsd	%xmm0, %xmm11
	addsd	%xmm11, %xmm10
	movsd	__real@3ff3333333333333(%rip), %xmm0
	movapd	%xmm7, %xmm1
	callq	pow
	mulsd	__real@3fe69d0369d0369d(%rip), %xmm0
	addsd	__real@3ff0000000000000(%rip), %xmm0
	mulsd	%xmm8, %xmm0
	addsd	%xmm10, %xmm0
	movapd	%xmm11, %xmm1
	mulsd	%xmm9, %xmm1
	divsd	%xmm0, %xmm1
	cmpl	$1, (%r13)
	movss	(%rdi), %xmm0
	cvtss2sd	%xmm0, %xmm0
	je	.LBB17_3
	mulsd	__real@3fe6666666666666(%rip), %xmm0
.LBB17_3:
	mulsd	__real@40204ec4ec4ec4ec(%rip), %xmm6
	addsd	%xmm6, %xmm10
	mulsd	__real@3fb16872b020c49c(%rip), %xmm11
	mulsd	%xmm0, %xmm11
	divsd	%xmm10, %xmm11
	movsd	%xmm11, (%r12)
	movsd	__real@3fe0000000000000(%rip), %xmm0
	mulsd	%xmm9, %xmm0
	movapd	%xmm11, %xmm2
	addsd	%xmm1, %xmm2
	movapd	%xmm2, %xmm3
	mulsd	%xmm2, %xmm3
	mulsd	__real@c00a8f5c28f5c28f(%rip), %xmm1
	mulsd	%xmm11, %xmm1
	addsd	%xmm3, %xmm1
	sqrtsd	%xmm1, %xmm1
	movapd	%xmm2, %xmm3
	subsd	%xmm1, %xmm3
	movsd	__real@3fe346f0940c565d(%rip), %xmm4
	mulsd	%xmm4, %xmm3
	addsd	%xmm1, %xmm2
	mulsd	%xmm4, %xmm2
	minsd	%xmm2, %xmm3
	addsd	%xmm3, %xmm0
	movapd	%xmm0, %xmm1
	mulsd	%xmm0, %xmm1
	mulsd	__real@bffdc28f5c28f5c3(%rip), %xmm9
	mulsd	%xmm3, %xmm9
.LBB17_7:
	addsd	%xmm1, %xmm9
	xorps	%xmm2, %xmm2
	sqrtsd	%xmm9, %xmm2
	movapd	%xmm0, %xmm1
	subsd	%xmm2, %xmm1
	addsd	%xmm2, %xmm0
	movsd	__real@3fe1344d1344d134(%rip), %xmm2
	mulsd	%xmm2, %xmm1
	mulsd	%xmm2, %xmm0
	minsd	%xmm0, %xmm1
	mulsd	__real@3feb333333333333(%rip), %xmm1
	xorpd	%xmm0, %xmm0
	maxsd	%xmm1, %xmm0
	movq	320(%rsp), %rax
	movsd	%xmm0, (%rax)
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	movaps	80(%rsp), %xmm9
	movaps	96(%rsp), %xmm10
	movaps	112(%rsp), %xmm11
	movaps	128(%rsp), %xmm12
	movaps	144(%rsp), %xmm13
	addq	$168, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
	.seh_endproc

	.def	PHOTO_mp_SPINUP3;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f800000
	.section	.rdata,"dr",discard,__real@3f800000
	.p2align	2, 0x0
__real@3f800000:
	.long	0x3f800000
	.globl	__real@3f28f5c3
	.section	.rdata,"dr",discard,__real@3f28f5c3
	.p2align	2, 0x0
__real@3f28f5c3:
	.long	0x3f28f5c3
	.globl	__real@3f800347
	.section	.rdata,"dr",discard,__real@3f800347
	.p2align	2, 0x0
__real@3f800347:
	.long	0x3f800347
	.text
	.globl	PHOTO_mp_SPINUP3
	.p2align	4, 0x90
PHOTO_mp_SPINUP3:
.seh_proc PHOTO_mp_SPINUP3
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$472, %rsp
	.seh_stackalloc 472
	movaps	%xmm15, 448(%rsp)
	.seh_savexmm %xmm15, 448
	movaps	%xmm14, 432(%rsp)
	.seh_savexmm %xmm14, 432
	movaps	%xmm13, 416(%rsp)
	.seh_savexmm %xmm13, 416
	movaps	%xmm12, 400(%rsp)
	.seh_savexmm %xmm12, 400
	movaps	%xmm11, 384(%rsp)
	.seh_savexmm %xmm11, 384
	movaps	%xmm10, 368(%rsp)
	.seh_savexmm %xmm10, 368
	movaps	%xmm9, 352(%rsp)
	.seh_savexmm %xmm9, 352
	movaps	%xmm8, 336(%rsp)
	.seh_savexmm %xmm8, 336
	movaps	%xmm7, 320(%rsp)
	.seh_savexmm %xmm7, 320
	movaps	%xmm6, 304(%rsp)
	.seh_savexmm %xmm6, 304
	.seh_endprologue
	movq	%r9, 80(%rsp)
	movq	%r8, 72(%rsp)
	movq	%rcx, %rsi
	movq	576(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	$0, 232(%rsp)
	movq	$0, 160(%rsp)
	movq	$0, 88(%rsp)
	movss	(%rdx), %xmm11
	movss	4(%rdx), %xmm7
	movss	8(%rdx), %xmm14
	movss	12(%rdx), %xmm6
	movss	16(%rdx), %xmm0
	movss	%xmm0, 32(%rsp)
	movss	20(%rdx), %xmm8
	movq	$0, 128(%rsp)
	movq	$4, 96(%rsp)
	movq	$1, 120(%rsp)
	movq	$0, 104(%rsp)
	movq	$1, 152(%rsp)
	movq	$65000, 136(%rsp)
	movq	$4, 144(%rsp)
	movq	$1073741957, 112(%rsp)
	leaq	88(%rsp), %rdx
	movl	$260000, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	movq	$0, 200(%rsp)
	movq	$4, 168(%rsp)
	movq	$1, 192(%rsp)
	movq	$0, 176(%rsp)
	movq	$1, 224(%rsp)
	movq	$65000, 208(%rsp)
	movq	$4, 216(%rsp)
	movq	$1073741957, 184(%rsp)
	leaq	160(%rsp), %rdx
	movl	$260000, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	movq	$0, 272(%rsp)
	movq	$4, 240(%rsp)
	movq	$1, 264(%rsp)
	movq	$0, 248(%rsp)
	movq	$1, 296(%rsp)
	movq	$65000, 280(%rsp)
	movq	$4, 288(%rsp)
	movq	$1073741957, 256(%rsp)
	leaq	232(%rsp), %rdx
	movl	$260000, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	movss	(%rsi), %xmm9
	xorps	%xmm10, %xmm10
	ucomiss	%xmm9, %xmm10
	movq	88(%rsp), %rbx
	movq	160(%rsp), %rdi
	movq	232(%rsp), %rsi
	jae	.LBB18_13
	movss	__real@3f800000(%rip), %xmm0
	movaps	%xmm0, %xmm12
	divss	%xmm11, %xmm12
	movaps	%xmm0, %xmm13
	divss	%xmm14, %xmm13
	divss	%xmm7, %xmm0
	movss	%xmm0, 56(%rsp)
	movq	152(%rsp), %r13
	mulss	%xmm9, %xmm6
	movq	224(%rsp), %rbp
	mulss	%xmm9, %xmm8
	movq	296(%rsp), %r14
	movl	$1, %eax
	movl	$1, %ecx
	subq	%r13, %rcx
	movq	%rcx, 40(%rsp)
	mulss	32(%rsp), %xmm9
	movl	$1, %r12d
	subq	%r14, %r12
	subq	%rbp, %rax
	movq	%rax, 48(%rsp)
	movl	$1, %r15d
	movss	__real@3f800347(%rip), %xmm15
	jmp	.LBB18_2
	.p2align	4, 0x90
.LBB18_3:
	movq	40(%rsp), %rax
	movss	%xmm6, (%rbx,%rax,4)
	movss	%xmm9, (%rsi,%r12,4)
	movq	48(%rsp), %rax
	movss	%xmm8, (%rdi,%rax,4)
	movl	$2, %r15d
.LBB18_2:
	cmpl	$1, %r15d
	je	.LBB18_3
	leal	-1(%r15), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	subq	%r13, %rax
	movq	%rcx, %rdx
	subq	%rbp, %rdx
	movss	32(%rsp), %xmm0
	ucomiss	%xmm10, %xmm0
	movss	(%rbx,%rax,4), %xmm0
	movaps	%xmm0, %xmm14
	addss	%xmm6, %xmm14
	movss	(%rdi,%rdx,4), %xmm1
	movaps	%xmm1, %xmm11
	addss	%xmm8, %xmm11
	mulss	%xmm12, %xmm0
	subss	%xmm0, %xmm14
	mulss	%xmm13, %xmm1
	subss	%xmm1, %xmm11
	movl	%r15d, %eax
	jbe	.LBB18_10
	subq	%r14, %rcx
	movss	(%rsi,%rcx,4), %xmm0
	movaps	%xmm0, %xmm7
	addss	%xmm6, %xmm7
	mulss	56(%rsp), %xmm0
	subss	%xmm0, %xmm7
	maxss	%xmm10, %xmm14
	movq	%rax, %rcx
	subq	%r13, %rcx
	movss	%xmm14, (%rbx,%rcx,4)
	maxss	%xmm10, %xmm7
	movq	%rax, %rcx
	subq	%r14, %rcx
	movss	%xmm7, (%rsi,%rcx,4)
	maxss	%xmm10, %xmm11
	subq	%rbp, %rax
	movss	%xmm11, (%rdi,%rax,4)
	xorps	%xmm0, %xmm0
	cvtsi2ss	%r15d, %xmm0
	mulss	__real@3f28f5c3(%rip), %xmm0
	callq	floorf
	cvttss2si	%xmm0, %eax
	cltq
	movq	%rax, %rcx
	subq	%rbp, %rcx
	movaps	%xmm11, %xmm0
	divss	(%rdi,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm15
	jbe	.LBB18_8
	movq	%rax, %rcx
	subq	%r13, %rcx
	movaps	%xmm14, %xmm0
	divss	(%rbx,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm15
	jbe	.LBB18_8
	subq	%r14, %rax
	movaps	%xmm7, %xmm0
	divss	(%rsi,%rax,4), %xmm0
	ucomiss	%xmm0, %xmm15
	jbe	.LBB18_8
	jmp	.LBB18_12
	.p2align	4, 0x90
.LBB18_10:
	xorps	%xmm7, %xmm7
	maxss	%xmm7, %xmm14
	movq	%rax, %rcx
	subq	%r13, %rcx
	movss	%xmm14, (%rbx,%rcx,4)
	maxss	%xmm7, %xmm11
	movq	%rax, %rcx
	subq	%rbp, %rcx
	movss	%xmm11, (%rdi,%rcx,4)
	subq	%r14, %rax
	movl	$0, (%rsi,%rax,4)
	xorps	%xmm0, %xmm0
	cvtsi2ss	%r15d, %xmm0
	mulss	__real@3f28f5c3(%rip), %xmm0
	callq	floorf
	cvttss2si	%xmm0, %eax
	cltq
	movq	%rax, %rcx
	subq	%rbp, %rcx
	movaps	%xmm11, %xmm0
	divss	(%rdi,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm15
	jbe	.LBB18_8
	subq	%r13, %rax
	movaps	%xmm14, %xmm0
	divss	(%rbx,%rax,4), %xmm0
	ucomiss	%xmm0, %xmm15
	ja	.LBB18_12
.LBB18_8:
	leal	1(%r15), %eax
	cmpl	$65000, %r15d
	movl	%eax, %r15d
	jb	.LBB18_2
	jmp	.LBB18_13
.LBB18_12:
	movq	72(%rsp), %rax
	movss	%xmm14, (%rax)
	movq	80(%rsp), %rax
	movss	%xmm11, (%rax)
	movq	64(%rsp), %rax
	movss	%xmm7, (%rax)
.LBB18_13:
	movq	112(%rsp), %r15
	movq	128(%rsp), %r8
	movl	%r15d, %eax
	andl	$3, %eax
	movl	%r15d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%r15, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	%rbx, %rcx
	movq	%r8, 32(%rsp)
	callq	for_dealloc_allocatable_handle
	movabsq	$-1030792153090, %r13
	movq	%r15, %r14
	andq	%r13, %r14
	movl	%eax, 40(%rsp)
	testl	%eax, %eax
	cmovneq	%r15, %r14
	movq	184(%rsp), %r15
	movq	200(%rsp), %r8
	movl	%r15d, %eax
	andl	$3, %eax
	movl	%r15d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%r15, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	%rdi, %rcx
	movq	%r8, 56(%rsp)
	callq	for_dealloc_allocatable_handle
	movq	%r15, %rbp
	andq	%r13, %rbp
	xorl	%r12d, %r12d
	testl	%eax, %eax
	cmoveq	%r12, %rdi
	cmovneq	%r15, %rbp
	movq	256(%rsp), %r15
	movq	272(%rsp), %r8
	movl	%r15d, %eax
	andl	$3, %eax
	movl	%r15d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%r15, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	%rsi, %rcx
	movq	%r8, 48(%rsp)
	callq	for_dealloc_allocatable_handle
	andq	%r15, %r13
	testl	%eax, %eax
	cmoveq	%r12, %rsi
	cmovneq	%r15, %r13
	movq	%r14, %rax
	andq	$1, %rax
	jne	.LBB18_14
	testb	$1, %bpl
	jne	.LBB18_16
.LBB18_17:
	testb	$1, %r13b
	jne	.LBB18_19
.LBB18_18:
	movaps	304(%rsp), %xmm6
	movaps	320(%rsp), %xmm7
	movaps	336(%rsp), %xmm8
	movaps	352(%rsp), %xmm9
	movaps	368(%rsp), %xmm10
	movaps	384(%rsp), %xmm11
	movaps	400(%rsp), %xmm12
	movaps	416(%rsp), %xmm13
	movaps	432(%rsp), %xmm14
	movaps	448(%rsp), %xmm15
	addq	$472, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
.LBB18_14:
	xorl	%ecx, %ecx
	cmpl	$0, 40(%rsp)
	cmoveq	%rcx, %rbx
	movl	%r14d, %ecx
	andl	$2, %ecx
	orl	%eax, %ecx
	movl	%r14d, %eax
	shrl	$3, %eax
	andl	$256, %eax
	leal	(%rax,%rcx,2), %eax
	shrq	$15, %r14
	andl	$65011712, %r14d
	leal	(%r14,%rax), %edx
	addl	$262144, %edx
	movq	%rbx, %rcx
	movq	32(%rsp), %r8
	callq	for_dealloc_allocatable_handle
	testb	$1, %bpl
	je	.LBB18_17
.LBB18_16:
	movl	%ebp, %eax
	andl	$2, %eax
	movl	%ebp, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	shrq	$15, %rbp
	andl	$65011712, %ebp
	leal	(%rcx,%rax,2), %eax
	leal	(%rax,%rbp), %edx
	addl	$262146, %edx
	movq	%rdi, %rcx
	movq	56(%rsp), %r8
	callq	for_dealloc_allocatable_handle
	testb	$1, %r13b
	je	.LBB18_18
.LBB18_19:
	movl	%r13d, %eax
	andl	$2, %eax
	movl	%r13d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	shrq	$15, %r13
	andl	$65011712, %r13d
	leal	(%rcx,%rax,2), %eax
	leal	(%rax,%r13), %edx
	addl	$262146, %edx
	movq	%rsi, %rcx
	movq	48(%rsp), %r8
	movaps	304(%rsp), %xmm6
	movaps	320(%rsp), %xmm7
	movaps	336(%rsp), %xmm8
	movaps	352(%rsp), %xmm9
	movaps	368(%rsp), %xmm10
	movaps	384(%rsp), %xmm11
	movaps	400(%rsp), %xmm12
	movaps	416(%rsp), %xmm13
	movaps	432(%rsp), %xmm14
	movaps	448(%rsp), %xmm15
	addq	$472, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	jmp	for_dealloc_allocatable_handle
	.seh_endproc

	.def	PHOTO_mp_SPINUP2;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f8147ae
	.section	.rdata,"dr",discard,__real@3f8147ae
	.p2align	2, 0x0
__real@3f8147ae:
	.long	0x3f8147ae
	.text
	.globl	PHOTO_mp_SPINUP2
	.p2align	4, 0x90
PHOTO_mp_SPINUP2:
.seh_proc PHOTO_mp_SPINUP2
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$504, %rsp
	.seh_stackalloc 504
	movaps	%xmm15, 480(%rsp)
	.seh_savexmm %xmm15, 480
	movaps	%xmm14, 464(%rsp)
	.seh_savexmm %xmm14, 464
	movaps	%xmm13, 448(%rsp)
	.seh_savexmm %xmm13, 448
	movaps	%xmm12, 432(%rsp)
	.seh_savexmm %xmm12, 432
	movaps	%xmm11, 416(%rsp)
	.seh_savexmm %xmm11, 416
	movaps	%xmm10, 400(%rsp)
	.seh_savexmm %xmm10, 400
	movaps	%xmm9, 384(%rsp)
	.seh_savexmm %xmm9, 384
	movaps	%xmm8, 368(%rsp)
	.seh_savexmm %xmm8, 368
	movaps	%xmm7, 352(%rsp)
	.seh_savexmm %xmm7, 352
	movaps	%xmm6, 336(%rsp)
	.seh_savexmm %xmm6, 336
	.seh_endprologue
	movq	%r9, 96(%rsp)
	movq	%r8, 88(%rsp)
	movq	%rdx, %r14
	movq	%rcx, %rbx
	movq	608(%rsp), %rax
	movq	%rax, 80(%rsp)
	movq	$0, 264(%rsp)
	movq	$0, 192(%rsp)
	movq	$0, 120(%rsp)
	movq	$0, 160(%rsp)
	movq	$4, 128(%rsp)
	movq	$1, 152(%rsp)
	movq	$0, 136(%rsp)
	movq	$1, 184(%rsp)
	movq	$36525, 168(%rsp)
	movq	$4, 176(%rsp)
	movq	$1073741957, 144(%rsp)
	leaq	120(%rsp), %rdx
	movl	$146100, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	movq	$0, 232(%rsp)
	movq	$4, 200(%rsp)
	movq	$1, 224(%rsp)
	movq	$0, 208(%rsp)
	movq	$1, 256(%rsp)
	movq	$36525, 240(%rsp)
	movq	$4, 248(%rsp)
	movq	$1073741957, 216(%rsp)
	leaq	192(%rsp), %rdx
	movl	$146100, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	movq	$0, 304(%rsp)
	movq	$4, 272(%rsp)
	movq	$1, 296(%rsp)
	movq	$0, 280(%rsp)
	movq	$1, 328(%rsp)
	movq	$36525, 312(%rsp)
	movq	$4, 320(%rsp)
	movq	$1073741957, 288(%rsp)
	leaq	264(%rsp), %rdx
	movl	$146100, %ecx
	movl	$262146, %r8d
	xorl	%r9d, %r9d
	callq	for_alloc_allocatable_handle
	leaq	8(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$TLEAF(%rip), %rdx
	.p2align	4, 0x90
.LBB19_1:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%rdx)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%rdx)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%rdx)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%rdx)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%rdx)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%rdx)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%rdx)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%rdx)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_1
	movss	67464(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3968(%rip)
	movss	67532(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3972(%rip)
	movss	67600(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3976(%rip)
	movss	67668(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3980(%rip)
	movss	67736(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3984(%rip)
	movss	67804(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3988(%rip)
	movss	67872(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TLEAF+3992(%rip)
	leaq	12(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$TAWOOD(%rip), %r8
	.p2align	4, 0x90
.LBB19_3:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%r8)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%r8)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%r8)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%r8)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%r8)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%r8)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%r8)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%r8)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_3
	movss	67468(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3968(%rip)
	movss	67536(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3972(%rip)
	movss	67604(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3976(%rip)
	movss	67672(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3980(%rip)
	movss	67740(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3984(%rip)
	movss	67808(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3988(%rip)
	movss	67876(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TAWOOD+3992(%rip)
	leaq	16(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$TFROOT(%rip), %r8
	.p2align	4, 0x90
.LBB19_5:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%r8)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%r8)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%r8)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%r8)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%r8)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%r8)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%r8)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%r8)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_5
	movss	67472(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3968(%rip)
	movss	67540(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3972(%rip)
	movss	67608(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3976(%rip)
	movss	67676(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3980(%rip)
	movss	67744(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3984(%rip)
	movss	67812(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3988(%rip)
	movss	67880(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$TFROOT+3992(%rip)
	leaq	20(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$ALEAF(%rip), %r9
	.p2align	4, 0x90
.LBB19_7:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%r9)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%r9)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%r9)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%r9)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%r9)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%r9)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%r9)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%r9)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_7
	movss	67476(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3968(%rip)
	movss	67544(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3972(%rip)
	movss	67612(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3976(%rip)
	movss	67680(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3980(%rip)
	movss	67748(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3984(%rip)
	movss	67816(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3988(%rip)
	movss	67884(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$ALEAF+3992(%rip)
	leaq	24(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$AAWOOD(%rip), %r10
	.p2align	4, 0x90
.LBB19_9:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%r10)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%r10)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%r10)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%r10)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%r10)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%r10)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%r10)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%r10)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_9
	movss	67480(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3968(%rip)
	movss	67548(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3972(%rip)
	movss	67616(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3976(%rip)
	movss	67684(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3980(%rip)
	movss	67752(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3984(%rip)
	movss	67820(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3988(%rip)
	movss	67888(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AAWOOD+3992(%rip)
	leaq	28(%r14), %rax
	movl	$28, %ecx
	leaq	PHOTO_mp_SPINUP2$AFROOT(%rip), %r11
	.p2align	4, 0x90
.LBB19_11:
	movss	(%rax), %xmm0
	movss	%xmm0, -28(%rcx,%r11)
	movss	68(%rax), %xmm0
	movss	%xmm0, -24(%rcx,%r11)
	movss	136(%rax), %xmm0
	movss	%xmm0, -20(%rcx,%r11)
	movss	204(%rax), %xmm0
	movss	%xmm0, -16(%rcx,%r11)
	movss	272(%rax), %xmm0
	movss	%xmm0, -12(%rcx,%r11)
	movss	340(%rax), %xmm0
	movss	%xmm0, -8(%rcx,%r11)
	movss	408(%rax), %xmm0
	movss	%xmm0, -4(%rcx,%r11)
	movss	476(%rax), %xmm0
	movss	%xmm0, (%rcx,%r11)
	addq	$544, %rax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB19_11
	movss	67484(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3968(%rip)
	movss	67552(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3972(%rip)
	movss	67620(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3976(%rip)
	movss	67688(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3980(%rip)
	movss	67756(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3984(%rip)
	movss	67824(%r14), %xmm0
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3988(%rip)
	movss	67892(%r14), %xmm0
	movss	(%rbx), %xmm6
	xorps	%xmm7, %xmm7
	ucomiss	%xmm6, %xmm7
	movss	%xmm0, PHOTO_mp_SPINUP2$AFROOT+3992(%rip)
	movq	120(%rsp), %r10
	movq	192(%rsp), %r14
	movq	264(%rsp), %rax
	movq	%rax, 48(%rsp)
	jae	.LBB19_27
	movq	184(%rsp), %rdi
	movq	256(%rsp), %r13
	movq	328(%rsp), %rax
	movl	$1, %esi
	movl	$1, %ecx
	subq	%rdi, %rcx
	movq	%rcx, 64(%rsp)
	movl	$1, %ecx
	movq	%rax, 40(%rsp)
	subq	%rax, %rcx
	movq	%rcx, 112(%rsp)
	movl	$1, %eax
	subq	%r13, %rax
	movq	%rax, 104(%rsp)
	movss	__real@3f800000(%rip), %xmm8
	movss	__real@3f28f5c3(%rip), %xmm9
	movss	__real@3f8147ae(%rip), %xmm10
	movq	%rdi, 56(%rsp)
	jmp	.LBB19_14
	.p2align	4, 0x90
.LBB19_25:
	movq	88(%rsp), %rax
	movss	%xmm14, -4(%rax,%rsi,4)
	movq	96(%rsp), %rax
	movss	%xmm15, -4(%rax,%rsi,4)
	movq	80(%rsp), %rax
	movss	%xmm13, -4(%rax,%rsi,4)
	movq	%rbp, %rdx
	movq	%r12, %r8
	leaq	PHOTO_mp_SPINUP2$ALEAF(%rip), %r9
	movq	%r15, %r11
.LBB19_26:
	incq	%rsi
	cmpq	$1000, %rsi
	je	.LBB19_27
.LBB19_14:
	leaq	PHOTO_mp_SPINUP2$AAWOOD(%rip), %rax
	movss	-4(%rax,%rsi,4), %xmm11
	leaq	PHOTO_mp_SPINUP2$TAWOOD(%rip), %rax
	movss	-4(%rax,%rsi,4), %xmm0
	xorps	%xmm1, %xmm1
	cmpltps	%xmm0, %xmm1
	xorps	%xmm2, %xmm2
	cmpltps	%xmm11, %xmm2
	andps	%xmm1, %xmm2
	movss	%xmm2, 72(%rsp)
	movaps	%xmm8, %xmm12
	divss	%xmm0, %xmm12
	mulss	%xmm6, %xmm11
	movl	$1, %ebx
	jmp	.LBB19_15
	.p2align	4, 0x90
.LBB19_16:
	movss	-4(%r9,%rsi,4), %xmm0
	mulss	%xmm6, %xmm0
	movq	64(%rsp), %rax
	movss	%xmm0, (%r10,%rax,4)
	movq	48(%rsp), %rax
	movq	112(%rsp), %rcx
	movss	%xmm11, (%rax,%rcx,4)
	movss	-4(%r11,%rsi,4), %xmm0
	mulss	%xmm6, %xmm0
	movq	104(%rsp), %rax
	movss	%xmm0, (%r14,%rax,4)
	movl	$2, %ebx
.LBB19_15:
	cmpl	$1, %ebx
	je	.LBB19_16
	leal	-1(%rbx), %eax
	movslq	%eax, %rcx
	movq	%rcx, %rax
	subq	%rdi, %rax
	movss	(%r10,%rax,4), %xmm0
	movss	-4(%r9,%rsi,4), %xmm13
	mulss	%xmm6, %xmm13
	movaps	%xmm13, %xmm14
	addss	%xmm0, %xmm14
	movq	%rcx, %rax
	subq	%r13, %rax
	movss	(%r14,%rax,4), %xmm1
	movq	%r11, %r15
	movss	-4(%r11,%rsi,4), %xmm15
	mulss	%xmm6, %xmm15
	addss	%xmm1, %xmm15
	movq	%rdx, %rbp
	divss	-4(%rdx,%rsi,4), %xmm0
	movq	%r8, %r12
	divss	-4(%r8,%rsi,4), %xmm1
	subss	%xmm0, %xmm14
	subss	%xmm1, %xmm15
	movl	%ebx, %eax
	testb	$1, 72(%rsp)
	je	.LBB19_21
	movq	40(%rsp), %r8
	subq	%r8, %rcx
	movq	48(%rsp), %rdx
	movss	(%rdx,%rcx,4), %xmm0
	addss	%xmm0, %xmm13
	mulss	%xmm12, %xmm0
	subss	%xmm0, %xmm13
	maxss	%xmm7, %xmm14
	movq	%rax, %rcx
	subq	%rdi, %rcx
	movss	%xmm14, (%r10,%rcx,4)
	maxss	%xmm7, %xmm13
	movq	%rax, %rcx
	subq	%r8, %rcx
	movss	%xmm13, (%rdx,%rcx,4)
	maxss	%xmm7, %xmm15
	subq	%r13, %rax
	movss	%xmm15, (%r14,%rax,4)
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ebx, %xmm0
	mulss	%xmm9, %xmm0
	movq	%r14, %rdi
	movq	%r10, %r14
	callq	floorf
	movq	%r14, %r10
	movq	%rdi, %r14
	movq	56(%rsp), %rdi
	cvttss2si	%xmm0, %eax
	cltq
	movq	%rax, %rcx
	subq	%r13, %rcx
	movaps	%xmm15, %xmm0
	divss	(%r14,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm10
	jbe	.LBB19_24
	movq	%rax, %rcx
	subq	%rdi, %rcx
	movaps	%xmm14, %xmm0
	divss	(%r10,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm10
	jbe	.LBB19_24
	subq	40(%rsp), %rax
	movaps	%xmm13, %xmm0
	movq	48(%rsp), %rcx
	divss	(%rcx,%rax,4), %xmm0
	ucomiss	%xmm0, %xmm10
	jbe	.LBB19_24
	jmp	.LBB19_25
	.p2align	4, 0x90
.LBB19_21:
	xorps	%xmm13, %xmm13
	maxss	%xmm13, %xmm14
	movq	%rax, %rcx
	subq	%rdi, %rcx
	movss	%xmm14, (%r10,%rcx,4)
	movq	%rax, %rcx
	subq	40(%rsp), %rcx
	movq	48(%rsp), %rdx
	movl	$0, (%rdx,%rcx,4)
	maxss	%xmm13, %xmm15
	subq	%r13, %rax
	movss	%xmm15, (%r14,%rax,4)
	xorps	%xmm0, %xmm0
	cvtsi2ss	%ebx, %xmm0
	mulss	%xmm9, %xmm0
	movq	%r14, %rdi
	movq	%r10, %r14
	callq	floorf
	movq	%r14, %r10
	movq	%rdi, %r14
	movq	56(%rsp), %rdi
	cvttss2si	%xmm0, %eax
	cltq
	movq	%rax, %rcx
	subq	%r13, %rcx
	movaps	%xmm15, %xmm0
	divss	(%r14,%rcx,4), %xmm0
	ucomiss	%xmm0, %xmm10
	jbe	.LBB19_24
	subq	%rdi, %rax
	movaps	%xmm14, %xmm0
	divss	(%r10,%rax,4), %xmm0
	ucomiss	%xmm0, %xmm10
	ja	.LBB19_25
.LBB19_24:
	leal	1(%rbx), %eax
	cmpl	$36525, %ebx
	movl	%eax, %ebx
	movq	%rbp, %rdx
	movq	%r12, %r8
	leaq	PHOTO_mp_SPINUP2$ALEAF(%rip), %r9
	movq	%r15, %r11
	jb	.LBB19_15
	jmp	.LBB19_26
.LBB19_27:
	movq	144(%rsp), %rsi
	movq	160(%rsp), %r8
	movl	%esi, %eax
	andl	$3, %eax
	movl	%esi, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%rsi, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	%r10, %rcx
	movq	%r8, 40(%rsp)
	movq	%r10, %r15
	callq	for_dealloc_allocatable_handle
	movabsq	$-1030792153090, %r13
	movq	%rsi, %rbx
	andq	%r13, %rbx
	movl	%eax, 64(%rsp)
	testl	%eax, %eax
	cmovneq	%rsi, %rbx
	movq	216(%rsp), %r12
	movq	232(%rsp), %r8
	movl	%r12d, %eax
	andl	$3, %eax
	movl	%r12d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%r12, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	%r14, %rcx
	movq	%r8, 72(%rsp)
	callq	for_dealloc_allocatable_handle
	movq	%r12, %rsi
	andq	%r13, %rsi
	xorl	%ebp, %ebp
	testl	%eax, %eax
	cmoveq	%rbp, %r14
	cmovneq	%r12, %rsi
	movq	288(%rsp), %r12
	movq	304(%rsp), %r8
	movl	%r12d, %eax
	andl	$3, %eax
	movl	%r12d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	movq	%r12, %rdx
	shrq	$15, %rdx
	andl	$65011712, %edx
	leal	(%rcx,%rax,2), %eax
	addl	%eax, %edx
	addl	$262144, %edx
	movq	48(%rsp), %rdi
	movq	%rdi, %rcx
	movq	%r8, 56(%rsp)
	callq	for_dealloc_allocatable_handle
	andq	%r12, %r13
	testl	%eax, %eax
	cmoveq	%rbp, %rdi
	cmovneq	%r12, %r13
	testb	$1, %bl
	jne	.LBB19_28
	testb	$1, %sil
	jne	.LBB19_30
.LBB19_31:
	testb	$1, %r13b
	jne	.LBB19_33
.LBB19_32:
	movaps	336(%rsp), %xmm6
	movaps	352(%rsp), %xmm7
	movaps	368(%rsp), %xmm8
	movaps	384(%rsp), %xmm9
	movaps	400(%rsp), %xmm10
	movaps	416(%rsp), %xmm11
	movaps	432(%rsp), %xmm12
	movaps	448(%rsp), %xmm13
	movaps	464(%rsp), %xmm14
	movaps	480(%rsp), %xmm15
	addq	$504, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
.LBB19_28:
	xorl	%eax, %eax
	cmpl	$0, 64(%rsp)
	cmoveq	%rax, %r15
	movl	%ebx, %eax
	andl	$2, %eax
	movl	%ebx, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	shrq	$15, %rbx
	andl	$65011712, %ebx
	leal	(%rcx,%rax,2), %eax
	leal	(%rbx,%rax), %edx
	addl	$262146, %edx
	movq	%r15, %rcx
	movq	40(%rsp), %r8
	callq	for_dealloc_allocatable_handle
	testb	$1, %sil
	je	.LBB19_31
.LBB19_30:
	movl	%esi, %eax
	andl	$2, %eax
	movl	%esi, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	shrq	$15, %rsi
	andl	$65011712, %esi
	leal	(%rcx,%rax,2), %eax
	leal	(%rsi,%rax), %edx
	addl	$262146, %edx
	movq	%r14, %rcx
	movq	72(%rsp), %r8
	callq	for_dealloc_allocatable_handle
	testb	$1, %r13b
	je	.LBB19_32
.LBB19_33:
	movl	%r13d, %eax
	andl	$2, %eax
	movl	%r13d, %ecx
	shrl	$3, %ecx
	andl	$256, %ecx
	shrq	$15, %r13
	andl	$65011712, %r13d
	leal	(%rcx,%rax,2), %eax
	leal	(%rax,%r13), %edx
	addl	$262146, %edx
	movq	%rdi, %rcx
	movq	56(%rsp), %r8
	movaps	336(%rsp), %xmm6
	movaps	352(%rsp), %xmm7
	movaps	368(%rsp), %xmm8
	movaps	384(%rsp), %xmm9
	movaps	400(%rsp), %xmm10
	movaps	416(%rsp), %xmm11
	movaps	432(%rsp), %xmm12
	movaps	448(%rsp), %xmm13
	movaps	464(%rsp), %xmm14
	movaps	480(%rsp), %xmm15
	addq	$504, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	jmp	for_dealloc_allocatable_handle
	.seh_endproc

	.def	PHOTO_mp_RESP_AUX;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@c2480000
	.section	.rdata,"dr",discard,__real@c2480000
	.p2align	2, 0x0
__real@c2480000:
	.long	0xc2480000
	.globl	__real@439f95c2
	.section	.rdata,"dr",discard,__real@439f95c2
	.p2align	2, 0x0
__real@439f95c2:
	.long	0x439f95c2
	.globl	__real@c39a47ae
	.section	.rdata,"dr",discard,__real@c39a47ae
	.p2align	2, 0x0
__real@c39a47ae:
	.long	0xc39a47ae
	.globl	__real@40b041ce
	.section	.rdata,"dr",discard,__real@40b041ce
	.p2align	2, 0x0
__real@40b041ce:
	.long	0x40b041ce
	.text
	.globl	PHOTO_mp_RESP_AUX
	.p2align	4, 0x90
PHOTO_mp_RESP_AUX:
	movss	(%rcx), %xmm1
	ucomiss	__real@c2480000(%rip), %xmm1
	jae	.LBB20_2
	xorps	%xmm0, %xmm0
	retq
.LBB20_2:
	addss	__real@439f95c2(%rip), %xmm1
	movss	__real@c39a47ae(%rip), %xmm0
	divss	%xmm1, %xmm0
	addss	__real@40b041ce(%rip), %xmm0
	jmp	expf

	.def	PHOTO_mp_F;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3d8d4fdf
	.section	.rdata,"dr",discard,__real@3d8d4fdf
	.p2align	2, 0x0
__real@3d8d4fdf:
	.long	0x3d8d4fdf
	.globl	__real@4196c75f
	.section	.rdata,"dr",discard,__real@4196c75f
	.p2align	2, 0x0
__real@4196c75f:
	.long	0x4196c75f
	.text
	.globl	PHOTO_mp_F
	.p2align	4, 0x90
PHOTO_mp_F:
	movss	(%rcx), %xmm0
	ucomiss	__real@c2480000(%rip), %xmm0
	jae	.LBB21_2
	xorps	%xmm0, %xmm0
	retq
.LBB21_2:
	mulss	__real@3d8d4fdf(%rip), %xmm0
	addss	__real@4196c75f(%rip), %xmm0
	jmp	expf

	.def	PHOTO_mp_M_RESP;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f73802a5ccadc32
	.section	.rdata,"dr",discard,__real@3f73802a5ccadc32
	.p2align	3, 0x0
__real@3f73802a5ccadc32:
	.quad	0x3f73802a5ccadc32
	.globl	__real@3fd24827b6fe2e6e
	.section	.rdata,"dr",discard,__real@3fd24827b6fe2e6e
	.p2align	3, 0x0
__real@3fd24827b6fe2e6e:
	.quad	0x3fd24827b6fe2e6e
	.globl	__real@3fd3802a5ccadc32
	.section	.rdata,"dr",discard,__real@3fd3802a5ccadc32
	.p2align	3, 0x0
__real@3fd3802a5ccadc32:
	.quad	0x3fd3802a5ccadc32
	.text
	.globl	PHOTO_mp_M_RESP
	.p2align	4, 0x90
PHOTO_mp_M_RESP:
.seh_proc PHOTO_mp_M_RESP
	pushq	%r14
	.seh_pushreg %r14
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$136, %rsp
	.seh_stackalloc 136
	movaps	%xmm11, 112(%rsp)
	.seh_savexmm %xmm11, 112
	movaps	%xmm10, 96(%rsp)
	.seh_savexmm %xmm10, 96
	movaps	%xmm9, 80(%rsp)
	.seh_savexmm %xmm9, 80
	movaps	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movapd	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movapd	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%r9, %rdi
	movq	%rdx, %rsi
	movq	216(%rsp), %rbx
	movq	240(%rsp), %rax
	movsd	(%rax), %xmm0
	xorpd	%xmm7, %xmm7
	ucomisd	%xmm7, %xmm0
	movss	(%rcx), %xmm10
	xorpd	%xmm6, %xmm6
	jbe	.LBB22_4
	movq	224(%rsp), %rax
	movq	208(%rsp), %rcx
	movsd	(%rcx), %xmm6
	mulsd	__real@3f73802a5ccadc32(%rip), %xmm6
	mulsd	(%rax), %xmm6
	xorpd	%xmm0, %xmm0
	ucomiss	__real@c2480000(%rip), %xmm10
	jb	.LBB22_3
	movss	__real@439f95c2(%rip), %xmm1
	addss	%xmm10, %xmm1
	movss	__real@c39a47ae(%rip), %xmm0
	divss	%xmm1, %xmm0
	addss	__real@40b041ce(%rip), %xmm0
	movq	%r8, %r14
	callq	expf
	movq	%r14, %r8
	cvtss2sd	%xmm0, %xmm0
.LBB22_3:
	mulsd	%xmm0, %xmm6
.LBB22_4:
	movq	232(%rsp), %r14
	movsd	(%r8), %xmm9
	movsd	(%rbx), %xmm8
	ucomiss	__real@c2480000(%rip), %xmm10
	jb	.LBB22_6
	addss	__real@439f95c2(%rip), %xmm10
	movss	__real@c39a47ae(%rip), %xmm0
	divss	%xmm10, %xmm0
	addss	__real@40b041ce(%rip), %xmm0
	callq	expf
	xorps	%xmm7, %xmm7
	cvtss2sd	%xmm0, %xmm7
.LBB22_6:
	movsd	(%rdi), %xmm10
	movsd	(%r14), %xmm11
	movss	(%rsi), %xmm1
	xorpd	%xmm0, %xmm0
	ucomiss	__real@c2480000(%rip), %xmm1
	jb	.LBB22_8
	addss	__real@439f95c2(%rip), %xmm1
	movss	__real@c39a47ae(%rip), %xmm0
	divss	%xmm1, %xmm0
	addss	__real@40b041ce(%rip), %xmm0
	callq	expf
	cvtss2sd	%xmm0, %xmm0
.LBB22_8:
	mulsd	%xmm0, %xmm11
	mulsd	__real@3fd24827b6fe2e6e(%rip), %xmm10
	mulsd	%xmm11, %xmm10
	mulsd	__real@3fd3802a5ccadc32(%rip), %xmm9
	mulsd	%xmm7, %xmm8
	mulsd	%xmm9, %xmm8
	addsd	%xmm6, %xmm8
	addsd	%xmm10, %xmm8
	xorps	%xmm1, %xmm1
	cvtsd2ss	%xmm8, %xmm1
	xorpd	%xmm0, %xmm0
	maxss	%xmm1, %xmm0
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	movaps	80(%rsp), %xmm9
	movaps	96(%rsp), %xmm10
	movaps	112(%rsp), %xmm11
	addq	$136, %rsp
	popq	%rbx
	popq	%rdi
	popq	%rsi
	popq	%r14
	retq
	.seh_endproc

	.def	PHOTO_mp_STO_RESP;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f6b4e81b4e81b4f
	.section	.rdata,"dr",discard,__real@3f6b4e81b4e81b4f
	.p2align	3, 0x0
__real@3f6b4e81b4e81b4f:
	.quad	0x3f6b4e81b4e81b4f
	.globl	__real@3fc86034f3fd933e
	.section	.rdata,"dr",discard,__real@3fc86034f3fd933e
	.p2align	3, 0x0
__real@3fc86034f3fd933e:
	.quad	0x3fc86034f3fd933e
	.text
	.globl	PHOTO_mp_STO_RESP
	.p2align	4, 0x90
PHOTO_mp_STO_RESP:
.seh_proc PHOTO_mp_STO_RESP
	subq	$56, %rsp
	.seh_stackalloc 56
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movsd	(%rdx), %xmm6
	xorpd	%xmm0, %xmm0
	ucomisd	%xmm6, %xmm0
	jae	.LBB23_7
	movsd	8(%rdx), %xmm1
	ucomisd	%xmm1, %xmm0
	jae	.LBB23_2
	divsd	%xmm6, %xmm1
	jmp	.LBB23_4
.LBB23_2:
	movsd	__real@3f6b4e81b4e81b4f(%rip), %xmm1
.LBB23_4:
	mulsd	__real@3fc86034f3fd933e(%rip), %xmm6
	mulsd	%xmm1, %xmm6
	movss	(%rcx), %xmm1
	ucomiss	__real@c2480000(%rip), %xmm1
	jb	.LBB23_6
	addss	__real@439f95c2(%rip), %xmm1
	movss	__real@c39a47ae(%rip), %xmm0
	divss	%xmm1, %xmm0
	addss	__real@40b041ce(%rip), %xmm0
	callq	expf
	cvtss2sd	%xmm0, %xmm0
.LBB23_6:
	mulsd	%xmm0, %xmm6
	xorpd	%xmm0, %xmm0
	maxsd	%xmm6, %xmm0
.LBB23_7:
	movaps	32(%rsp), %xmm6
	addq	$56, %rsp
	retq
	.seh_endproc

	.def	PHOTO_mp_G_RESP;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f30624dd2f1a9fc
	.section	.rdata,"dr",discard,__real@3f30624dd2f1a9fc
	.p2align	3, 0x0
__real@3f30624dd2f1a9fc:
	.quad	0x3f30624dd2f1a9fc
	.text
	.globl	PHOTO_mp_G_RESP
	.p2align	4, 0x90
PHOTO_mp_G_RESP:
	movsd	(%rcx), %xmm0
	mulsd	__real@3f30624dd2f1a9fc(%rip), %xmm0
	cvtsd2ss	%xmm0, %xmm0
	retq

	.def	PHOTO_mp_TETENS;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_mp_TETENS
	.p2align	4, 0x90
PHOTO_mp_TETENS:
.seh_proc PHOTO_mp_TETENS
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$32, %rsp
	.seh_stackalloc 32
	.seh_endprologue
	movss	(%rcx), %xmm1
	xorps	%xmm0, %xmm0
	xorl	%esi, %esi
	ucomiss	%xmm0, %xmm1
	setb	%sil
	leaq	__real@41b849ba4195d4fe(%rip), %rax
	movss	(%rax,%rsi,4), %xmm0
	leaq	__real@3b4464583b900901(%rip), %rax
	movss	(%rax,%rsi,4), %xmm2
	mulss	%xmm1, %xmm2
	subss	%xmm2, %xmm0
	leaq	__real@438be8f64380ef5c(%rip), %rax
	mulss	%xmm1, %xmm0
	addss	(%rax,%rsi,4), %xmm1
	divss	%xmm1, %xmm0
	callq	expf
	leaq	__real@40c3916840c39653(%rip), %rax
	mulss	(%rax,%rsi,4), %xmm0
	addq	$32, %rsp
	popq	%rsi
	retq
	.seh_endproc

	.def	PHOTO_mp_PFT_AREA_FRAC;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3f50624dd2f1a9fc
	.section	.rdata,"dr",discard,__real@3f50624dd2f1a9fc
	.p2align	3, 0x0
__real@3f50624dd2f1a9fc:
	.quad	0x3f50624dd2f1a9fc
	.globl	__xmm@00ff00ff00ff00ff00ff00ff00ff00ff
	.section	.rdata,"dr",discard,__xmm@00ff00ff00ff00ff00ff00ff00ff00ff
	.p2align	4, 0x0
__xmm@00ff00ff00ff00ff00ff00ff00ff00ff:
	.short	255
	.short	255
	.short	255
	.short	255
	.short	255
	.short	255
	.short	255
	.short	255
	.globl	__xmm@00000001000000010000000100000001
	.section	.rdata,"dr",discard,__xmm@00000001000000010000000100000001
	.p2align	4, 0x0
__xmm@00000001000000010000000100000001:
	.long	1
	.long	1
	.long	1
	.long	1
	.globl	__real@3d4ccccd
	.section	.rdata,"dr",discard,__real@3d4ccccd
	.p2align	2, 0x0
__real@3d4ccccd:
	.long	0x3d4ccccd
	.globl	__real@fff0000000000000
	.section	.rdata,"dr",discard,__real@fff0000000000000
	.p2align	3, 0x0
__real@fff0000000000000:
	.quad	0xfff0000000000000
	.text
	.globl	PHOTO_mp_PFT_AREA_FRAC
	.p2align	4, 0x90
PHOTO_mp_PFT_AREA_FRAC:
.seh_proc PHOTO_mp_PFT_AREA_FRAC
	pushq	%r15
	.seh_pushreg %r15
	pushq	%r14
	.seh_pushreg %r14
	pushq	%r13
	.seh_pushreg %r13
	pushq	%r12
	.seh_pushreg %r12
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbp
	.seh_pushreg %rbp
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$184, %rsp
	.seh_stackalloc 184
	movapd	%xmm14, 160(%rsp)
	.seh_savexmm %xmm14, 160
	movdqa	%xmm13, 144(%rsp)
	.seh_savexmm %xmm13, 144
	movapd	%xmm12, 128(%rsp)
	.seh_savexmm %xmm12, 128
	movdqa	%xmm11, 112(%rsp)
	.seh_savexmm %xmm11, 112
	movapd	%xmm10, 96(%rsp)
	.seh_savexmm %xmm10, 96
	movdqa	%xmm9, 80(%rsp)
	.seh_savexmm %xmm9, 80
	movdqa	%xmm8, 64(%rsp)
	.seh_savexmm %xmm8, 64
	movaps	%xmm7, 48(%rsp)
	.seh_savexmm %xmm7, 48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%r9, %r15
	movq	%r8, %rdi
	movq	%rdx, %rbp
	movq	%rcx, %rdx
	movq	304(%rsp), %rbx
	leaq	PHOTO_mp_PFT_AREA_FRAC$CLEAF(%rip), %r12
	movl	$7992, %r8d
	movq	%r12, %rcx
	callq	_intel_fast_memcpy
	leaq	PHOTO_mp_PFT_AREA_FRAC$CFROOT(%rip), %r13
	movl	$7992, %r8d
	movq	%r13, %rcx
	movq	%rbp, %rdx
	callq	_intel_fast_memcpy
	movl	$56, %eax
	xorpd	%xmm0, %xmm0
	leaq	PHOTO_mp_PFT_AREA_FRAC$CAWOOD(%rip), %rsi
	.p2align	4, 0x90
.LBB26_1:
	movsd	-56(%r15,%rax), %xmm1
	movsd	-56(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -56(%rax,%rsi)
	movsd	-48(%r15,%rax), %xmm1
	movsd	-48(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -48(%rax,%rsi)
	movsd	-40(%r15,%rax), %xmm1
	movsd	-40(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -40(%rax,%rsi)
	movsd	-32(%r15,%rax), %xmm1
	movsd	-32(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -32(%rax,%rsi)
	movsd	-24(%r15,%rax), %xmm1
	movsd	-24(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -24(%rax,%rsi)
	movsd	-16(%r15,%rax), %xmm1
	movsd	-16(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -16(%rax,%rsi)
	movsd	-8(%r15,%rax), %xmm1
	movsd	-8(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, -8(%rax,%rsi)
	movsd	(%r15,%rax), %xmm1
	movsd	(%rdi,%rax), %xmm2
	cmplesd	%xmm0, %xmm1
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, (%rax,%rsi)
	addq	$64, %rax
	cmpq	$7992, %rax
	jne	.LBB26_1
	movsd	7936(%r15), %xmm1
	xorpd	%xmm0, %xmm0
	cmplesd	%xmm0, %xmm1
	movsd	7936(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7936(%rip)
	movsd	7944(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7944(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7944(%rip)
	movsd	7952(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7952(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7952(%rip)
	movsd	7960(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7960(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7960(%rip)
	movsd	7968(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7968(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7968(%rip)
	movsd	7976(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7976(%rdi), %xmm2
	andnpd	%xmm2, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7976(%rip)
	movsd	7984(%r15), %xmm1
	cmplesd	%xmm0, %xmm1
	movsd	7984(%rdi), %xmm0
	andnpd	%xmm0, %xmm1
	movlpd	%xmm1, PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7984(%rip)
	leaq	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT(%rip), %rdi
	xorl	%r14d, %r14d
	movl	$7992, %r8d
	movq	%rdi, %rcx
	xorl	%edx, %edx
	callq	_intel_fast_memset
	leaq	PHOTO_mp_PFT_AREA_FRAC$TOTAL_BIOMASS_PFT(%rip), %r15
	movl	$7992, %r8d
	movq	%r15, %rcx
	xorl	%edx, %edx
	callq	_intel_fast_memset
	movl	$7992, %r8d
	movq	288(%rsp), %rcx
	xorl	%edx, %edx
	callq	_intel_fast_memset
	movl	$3996, %r8d
	movq	296(%rsp), %rcx
	xorl	%edx, %edx
	callq	_intel_fast_memset
	jmp	.LBB26_3
	.p2align	4, 0x90
.LBB26_9:
	addq	$8, %r14
	cmpq	$7992, %r14
	je	.LBB26_10
.LBB26_3:
	leaq	(%r12,%r14), %rbp
	movq	%rbp, %rcx
	callq	for_is_nan_t_
	testb	$1, %al
	je	.LBB26_5
	movq	$0, (%rbp)
.LBB26_5:
	leaq	(%r14,%r13), %rbp
	movq	%rbp, %rcx
	callq	for_is_nan_t_
	testb	$1, %al
	je	.LBB26_7
	movq	$0, (%rbp)
.LBB26_7:
	leaq	(%rsi,%r14), %rbp
	movq	%rbp, %rcx
	callq	for_is_nan_t_
	testb	$1, %al
	je	.LBB26_9
	movq	$0, (%rbp)
	jmp	.LBB26_9
.LBB26_10:
	leaq	PHOTO_mp_PFT_AREA_FRAC$IS_LIVING(%rip), %rax
	xorl	%r8d, %r8d
	movsd	__real@3f50624dd2f1a9fc(%rip), %xmm0
	movq	288(%rsp), %rcx
	movq	312(%rsp), %r9
	jmp	.LBB26_11
	.p2align	4, 0x90
.LBB26_14:
	movsd	%xmm1, (%r9,%r8)
	movw	%dx, (%rax)
	addq	$8, %r8
	addq	$2, %rax
	cmpq	$7992, %r8
	je	.LBB26_15
.LBB26_11:
	movsd	(%r8,%r12), %xmm3
	ucomisd	%xmm3, %xmm0
	movw	$-1, %dx
	xorpd	%xmm1, %xmm1
	jbe	.LBB26_14
	movsd	(%r8,%r13), %xmm2
	ucomisd	%xmm2, %xmm0
	jbe	.LBB26_14
	addsd	%xmm3, %xmm2
	addsd	(%r8,%rsi), %xmm2
	movq	$0, (%r8,%r12)
	movq	$0, (%r8,%rsi)
	movq	$0, (%r8,%r13)
	xorl	%edx, %edx
	movapd	%xmm2, %xmm1
	jmp	.LBB26_14
.LBB26_15:
	xorpd	%xmm0, %xmm0
	movq	$-2, %rax
	xorl	%edx, %edx
	xorpd	%xmm1, %xmm1
	.p2align	4, 0x90
.LBB26_16:
	movapd	(%rdx,%r13), %xmm2
	addpd	(%rdx,%r12), %xmm2
	movapd	(%rdx,%rsi), %xmm3
	addpd	%xmm3, %xmm2
	movapd	%xmm2, (%rdx,%r15)
	addpd	%xmm2, %xmm0
	addpd	%xmm3, %xmm1
	movapd	%xmm3, (%rdx,%rdi)
	addq	$2, %rax
	addq	$16, %rdx
	cmpq	$996, %rax
	jb	.LBB26_16
	movsd	PHOTO_mp_PFT_AREA_FRAC$CFROOT+7984(%rip), %xmm3
	addsd	PHOTO_mp_PFT_AREA_FRAC$CLEAF+7984(%rip), %xmm3
	movsd	PHOTO_mp_PFT_AREA_FRAC$CAWOOD+7984(%rip), %xmm6
	addsd	%xmm6, %xmm3
	movsd	%xmm3, PHOTO_mp_PFT_AREA_FRAC$TOTAL_BIOMASS_PFT+7984(%rip)
	movapd	%xmm0, %xmm2
	unpckhpd	%xmm0, %xmm2
	addsd	%xmm0, %xmm2
	addsd	%xmm3, %xmm2
	movapd	%xmm1, %xmm7
	unpckhpd	%xmm1, %xmm7
	addsd	%xmm1, %xmm7
	movsd	%xmm6, PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7984(%rip)
	xorpd	%xmm0, %xmm0
	ucomisd	%xmm0, %xmm2
	jbe	.LBB26_18
	movsd	__real@3ff0000000000000(%rip), %xmm0
	divsd	%xmm2, %xmm0
	movapd	%xmm0, %xmm1
	unpcklpd	%xmm0, %xmm1
	movq	$-16, %rax
	leaq	PHOTO_mp_PFT_AREA_FRAC$IS_LIVING(%rip), %rdx
	movdqa	__xmm@00ff00ff00ff00ff00ff00ff00ff00ff(%rip), %xmm3
	movdqa	__xmm@00000001000000010000000100000001(%rip), %xmm2
	movq	296(%rsp), %r12
	.p2align	4, 0x90
.LBB26_32:
	movapd	64(%r15), %xmm8
	mulpd	%xmm1, %xmm8
	movapd	80(%r15), %xmm9
	mulpd	%xmm1, %xmm9
	movapd	32(%r15), %xmm10
	mulpd	%xmm1, %xmm10
	movapd	48(%r15), %xmm11
	mulpd	%xmm1, %xmm11
	movapd	(%r15), %xmm12
	mulpd	%xmm1, %xmm12
	movapd	16(%r15), %xmm13
	mulpd	%xmm1, %xmm13
	movapd	96(%r15), %xmm14
	mulpd	%xmm1, %xmm14
	movapd	112(%r15), %xmm5
	mulpd	%xmm1, %xmm5
	xorpd	%xmm4, %xmm4
	maxpd	%xmm5, %xmm4
	xorpd	%xmm5, %xmm5
	maxpd	%xmm14, %xmm5
	xorpd	%xmm14, %xmm14
	maxpd	%xmm13, %xmm14
	xorpd	%xmm13, %xmm13
	maxpd	%xmm12, %xmm13
	xorpd	%xmm12, %xmm12
	maxpd	%xmm11, %xmm12
	xorpd	%xmm11, %xmm11
	maxpd	%xmm10, %xmm11
	xorpd	%xmm10, %xmm10
	maxpd	%xmm9, %xmm10
	xorpd	%xmm9, %xmm9
	maxpd	%xmm8, %xmm9
	movupd	%xmm9, 192(%rcx,%rax,8)
	movupd	%xmm10, 208(%rcx,%rax,8)
	movupd	%xmm11, 160(%rcx,%rax,8)
	movupd	%xmm12, 176(%rcx,%rax,8)
	movupd	%xmm13, 128(%rcx,%rax,8)
	movupd	%xmm14, 144(%rcx,%rax,8)
	movupd	%xmm5, 224(%rcx,%rax,8)
	movupd	%xmm4, 240(%rcx,%rax,8)
	xorpd	%xmm8, %xmm8
	cmpltpd	%xmm13, %xmm8
	xorpd	%xmm13, %xmm13
	cmpltpd	%xmm14, %xmm13
	packssdw	%xmm13, %xmm8
	xorpd	%xmm13, %xmm13
	cmpltpd	%xmm11, %xmm13
	xorpd	%xmm11, %xmm11
	cmpltpd	%xmm12, %xmm11
	packssdw	%xmm11, %xmm13
	packssdw	%xmm13, %xmm8
	xorpd	%xmm11, %xmm11
	cmpltpd	%xmm9, %xmm11
	xorpd	%xmm9, %xmm9
	cmpltpd	%xmm10, %xmm9
	packssdw	%xmm9, %xmm11
	xorpd	%xmm9, %xmm9
	cmpltpd	%xmm5, %xmm9
	xorpd	%xmm5, %xmm5
	cmpltpd	%xmm4, %xmm5
	packssdw	%xmm5, %xmm9
	packssdw	%xmm9, %xmm11
	packsswb	%xmm11, %xmm8
	movdqa	16(%rdx), %xmm4
	pand	%xmm3, %xmm4
	movdqa	(%rdx), %xmm5
	pand	%xmm3, %xmm5
	packuswb	%xmm4, %xmm5
	pand	%xmm8, %xmm5
	movdqa	%xmm5, %xmm4
	punpcklbw	%xmm4, %xmm4
	movdqa	%xmm4, %xmm8
	punpcklwd	%xmm8, %xmm8
	pand	%xmm2, %xmm8
	punpckhwd	%xmm4, %xmm4
	pand	%xmm2, %xmm4
	punpckhbw	%xmm5, %xmm5
	movdqa	%xmm5, %xmm9
	punpcklwd	%xmm9, %xmm9
	pand	%xmm2, %xmm9
	punpckhwd	%xmm5, %xmm5
	pand	%xmm2, %xmm5
	movdqu	%xmm5, 112(%rbx,%rax,4)
	movdqu	%xmm9, 96(%rbx,%rax,4)
	movdqu	%xmm4, 80(%rbx,%rax,4)
	movdqu	%xmm8, 64(%rbx,%rax,4)
	addq	$16, %rax
	addq	$32, %rdx
	subq	$-128, %r15
	cmpq	$976, %rax
	jb	.LBB26_32
	movl	$988, %edx
	leaq	PHOTO_mp_PFT_AREA_FRAC$TOTAL_BIOMASS_PFT(%rip), %rax
	leaq	PHOTO_mp_PFT_AREA_FRAC$IS_LIVING(%rip), %r9
	.p2align	4, 0x90
.LBB26_34:
	movapd	32(%rax,%rdx,8), %xmm3
	mulpd	%xmm1, %xmm3
	movapd	48(%rax,%rdx,8), %xmm4
	mulpd	%xmm1, %xmm4
	pxor	%xmm5, %xmm5
	maxpd	%xmm4, %xmm5
	xorpd	%xmm4, %xmm4
	maxpd	%xmm3, %xmm4
	movupd	%xmm4, 32(%rcx,%rdx,8)
	movupd	%xmm5, 48(%rcx,%rdx,8)
	movq	8(%r9,%rdx,2), %xmm3
	punpcklwd	%xmm3, %xmm3
	pxor	%xmm8, %xmm8
	cmpltpd	%xmm4, %xmm8
	xorpd	%xmm4, %xmm4
	cmpltpd	%xmm5, %xmm4
	shufps	$136, %xmm4, %xmm8
	andps	%xmm3, %xmm8
	andps	%xmm2, %xmm8
	movups	%xmm8, 16(%rbx,%rdx,4)
	addq	$4, %rdx
	cmpq	$992, %rdx
	jb	.LBB26_34
	xorl	%edx, %edx
	xorpd	%xmm1, %xmm1
	.p2align	4, 0x90
.LBB26_36:
	movsd	7968(%rax,%rdx,4), %xmm2
	mulsd	%xmm0, %xmm2
	pxor	%xmm3, %xmm3
	maxsd	%xmm2, %xmm3
	movsd	%xmm3, 7968(%rcx,%rdx,4)
	ucomisd	%xmm1, %xmm3
	seta	%r8b
	andb	1992(%rdx,%r9), %r8b
	movzbl	%r8b, %r8d
	movl	%r8d, 3984(%rbx,%rdx,2)
	addq	$2, %rdx
	cmpq	$6, %rdx
	jne	.LBB26_36
	jmp	.LBB26_37
.LBB26_18:
	movl	$7992, %r8d
	xorl	%edx, %edx
	callq	_intel_fast_memset
	movl	$3996, %r8d
	movq	%rbx, %rcx
	xorl	%edx, %edx
	callq	_intel_fast_memset
	movq	296(%rsp), %r12
.LBB26_37:
	addsd	%xmm7, %xmm6
	xorl	%eax, %eax
	movl	$28, %ecx
	.p2align	4, 0x90
.LBB26_38:
	addl	-28(%rbx,%rcx), %eax
	addl	-24(%rbx,%rcx), %eax
	addl	-20(%rbx,%rcx), %eax
	addl	-16(%rbx,%rcx), %eax
	addl	-12(%rbx,%rcx), %eax
	addl	-8(%rbx,%rcx), %eax
	addl	-4(%rbx,%rcx), %eax
	addl	(%rbx,%rcx), %eax
	addq	$32, %rcx
	cmpq	$3996, %rcx
	jne	.LBB26_38
	addl	3968(%rbx), %eax
	addl	3972(%rbx), %eax
	addl	3976(%rbx), %eax
	addl	3980(%rbx), %eax
	addl	3984(%rbx), %eax
	addl	3988(%rbx), %eax
	addl	3992(%rbx), %eax
	xorps	%xmm0, %xmm0
	cvtsi2ss	%eax, %xmm0
	mulss	__real@3d4ccccd(%rip), %xmm0
	callq	roundf
	cvttss2si	%xmm0, %eax
	cmpl	$2, %eax
	jae	.LBB26_23
	xorps	%xmm0, %xmm0
	ucomisd	%xmm0, %xmm6
	jbe	.LBB26_30
	movsd	__real@fff0000000000000(%rip), %xmm0
	movl	$1, %ecx
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB26_21:
	leal	1(%rax), %edx
	movsd	(%rdi,%rax,8), %xmm1
	ucomisd	%xmm0, %xmm1
	movsd	8(%rdi,%rax,8), %xmm2
	cmovbel	%ecx, %edx
	maxsd	%xmm0, %xmm1
	leal	2(%rax), %ecx
	ucomisd	%xmm1, %xmm2
	cmovbel	%edx, %ecx
	maxsd	%xmm1, %xmm2
	movsd	16(%rdi,%rax,8), %xmm0
	leal	3(%rax), %edx
	ucomisd	%xmm2, %xmm0
	cmovbel	%ecx, %edx
	maxsd	%xmm2, %xmm0
	movsd	24(%rdi,%rax,8), %xmm1
	leal	4(%rax), %ecx
	ucomisd	%xmm0, %xmm1
	cmovbel	%edx, %ecx
	maxsd	%xmm0, %xmm1
	movsd	32(%rdi,%rax,8), %xmm0
	leal	5(%rax), %edx
	ucomisd	%xmm1, %xmm0
	cmovbel	%ecx, %edx
	maxsd	%xmm1, %xmm0
	movsd	40(%rdi,%rax,8), %xmm1
	leal	6(%rax), %ecx
	ucomisd	%xmm0, %xmm1
	cmovbel	%edx, %ecx
	maxsd	%xmm0, %xmm1
	movsd	48(%rdi,%rax,8), %xmm2
	leal	7(%rax), %edx
	ucomisd	%xmm1, %xmm2
	cmovbel	%ecx, %edx
	maxsd	%xmm1, %xmm2
	movsd	56(%rdi,%rax,8), %xmm0
	addq	$8, %rax
	ucomisd	%xmm2, %xmm0
	movl	%eax, %ecx
	cmovbel	%edx, %ecx
	maxsd	%xmm2, %xmm0
	cmpq	$992, %rax
	jne	.LBB26_21
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7936(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	movl	$993, %eax
	cmoval	%eax, %ecx
	maxsd	%xmm0, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7944(%rip), %xmm0
	ucomisd	%xmm1, %xmm0
	movl	$994, %eax
	cmovbel	%ecx, %eax
	maxsd	%xmm1, %xmm0
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7952(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	movl	$995, %ecx
	cmovbel	%eax, %ecx
	maxsd	%xmm0, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7960(%rip), %xmm0
	ucomisd	%xmm1, %xmm0
	movl	$996, %eax
	cmovbel	%ecx, %eax
	maxsd	%xmm1, %xmm0
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7968(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	movl	$997, %ecx
	cmovbel	%eax, %ecx
	maxsd	%xmm0, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7976(%rip), %xmm0
	ucomisd	%xmm1, %xmm0
	movl	$998, %eax
	cmovbel	%ecx, %eax
	maxsd	%xmm1, %xmm0
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7984(%rip), %xmm1
	ucomisd	%xmm0, %xmm1
	movl	$999, %ecx
	cmovbel	%eax, %ecx
	movl	%ecx, PHOTO_mp_PFT_AREA_FRAC$MAX_INDEX(%rip)
	movslq	%ecx, %rax
	movl	$1, -4(%r12,%rax,4)
	jmp	.LBB26_30
.LBB26_23:
	testl	%eax, %eax
	jle	.LBB26_30
	xorps	%xmm0, %xmm0
	ucomisd	%xmm0, %xmm6
	jbe	.LBB26_30
	decl	%eax
	xorl	%ecx, %ecx
	movsd	__real@fff0000000000000(%rip), %xmm0
	movl	$993, %edx
	movl	$994, %r8d
	movl	$995, %r9d
	movl	$996, %r10d
	movl	$997, %r11d
	movl	$998, %esi
	movl	$999, %ebx
	.p2align	4, 0x90
.LBB26_26:
	movl	$1, %ebp
	xorl	%r14d, %r14d
	movapd	%xmm0, %xmm1
	.p2align	4, 0x90
.LBB26_27:
	leal	1(%r14), %r15d
	movsd	(%rdi,%r14,8), %xmm2
	ucomisd	%xmm1, %xmm2
	movsd	8(%rdi,%r14,8), %xmm3
	cmovbel	%ebp, %r15d
	maxsd	%xmm1, %xmm2
	leal	2(%r14), %ebp
	ucomisd	%xmm2, %xmm3
	cmovbel	%r15d, %ebp
	maxsd	%xmm2, %xmm3
	movsd	16(%rdi,%r14,8), %xmm1
	leal	3(%r14), %r15d
	ucomisd	%xmm3, %xmm1
	cmovbel	%ebp, %r15d
	maxsd	%xmm3, %xmm1
	movsd	24(%rdi,%r14,8), %xmm2
	leal	4(%r14), %ebp
	ucomisd	%xmm1, %xmm2
	cmovbel	%r15d, %ebp
	maxsd	%xmm1, %xmm2
	movsd	32(%rdi,%r14,8), %xmm1
	leal	5(%r14), %r15d
	ucomisd	%xmm2, %xmm1
	cmovbel	%ebp, %r15d
	maxsd	%xmm2, %xmm1
	movsd	40(%rdi,%r14,8), %xmm2
	leal	6(%r14), %ebp
	ucomisd	%xmm1, %xmm2
	cmovbel	%r15d, %ebp
	maxsd	%xmm1, %xmm2
	movsd	48(%rdi,%r14,8), %xmm3
	leal	7(%r14), %r15d
	ucomisd	%xmm2, %xmm3
	cmovbel	%ebp, %r15d
	maxsd	%xmm2, %xmm3
	movsd	56(%rdi,%r14,8), %xmm1
	addq	$8, %r14
	ucomisd	%xmm3, %xmm1
	movl	%r14d, %ebp
	cmovbel	%r15d, %ebp
	maxsd	%xmm3, %xmm1
	cmpq	$992, %r14
	jne	.LBB26_27
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7936(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	cmoval	%edx, %ebp
	maxsd	%xmm1, %xmm2
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7944(%rip), %xmm1
	ucomisd	%xmm2, %xmm1
	cmoval	%r8d, %ebp
	maxsd	%xmm2, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7952(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	cmoval	%r9d, %ebp
	maxsd	%xmm1, %xmm2
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7960(%rip), %xmm1
	ucomisd	%xmm2, %xmm1
	cmoval	%r10d, %ebp
	maxsd	%xmm2, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7968(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	cmoval	%r11d, %ebp
	maxsd	%xmm1, %xmm2
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7976(%rip), %xmm1
	ucomisd	%xmm2, %xmm1
	cmoval	%esi, %ebp
	maxsd	%xmm2, %xmm1
	movsd	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT+7984(%rip), %xmm2
	ucomisd	%xmm1, %xmm2
	cmoval	%ebx, %ebp
	movslq	%ebp, %r14
	movq	$0, -8(%rdi,%r14,8)
	movl	$1, -4(%r12,%r14,4)
	leal	1(%rcx), %r14d
	cmpl	%eax, %ecx
	movl	%r14d, %ecx
	jne	.LBB26_26
	movl	%ebp, PHOTO_mp_PFT_AREA_FRAC$MAX_INDEX(%rip)
.LBB26_30:
	movaps	32(%rsp), %xmm6
	movaps	48(%rsp), %xmm7
	movaps	64(%rsp), %xmm8
	movaps	80(%rsp), %xmm9
	movaps	96(%rsp), %xmm10
	movaps	112(%rsp), %xmm11
	movaps	128(%rsp), %xmm12
	movaps	144(%rsp), %xmm13
	movaps	160(%rsp), %xmm14
	addq	$184, %rsp
	popq	%rbx
	popq	%rbp
	popq	%rdi
	popq	%rsi
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	retq
	.seh_endproc

	.def	PHOTO_mp_VEC_RANGING;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@ff7fffff
	.section	.rdata,"dr",discard,__real@ff7fffff
	.p2align	2, 0x0
__real@ff7fffff:
	.long	0xff7fffff
	.globl	__real@7f7fffff
	.section	.rdata,"dr",discard,__real@7f7fffff
	.p2align	2, 0x0
__real@7f7fffff:
	.long	0x7f7fffff
	.globl	__real@7f800000
	.section	.rdata,"dr",discard,__real@7f800000
	.p2align	2, 0x0
__real@7f800000:
	.long	0x7f800000
	.globl	__real@ff800000
	.section	.rdata,"dr",discard,__real@ff800000
	.p2align	2, 0x0
__real@ff800000:
	.long	0xff800000
	.globl	__xmm@00000000000000030000000000000002
	.section	.rdata,"dr",discard,__xmm@00000000000000030000000000000002
	.p2align	4, 0x0
__xmm@00000000000000030000000000000002:
	.quad	2
	.quad	3
	.globl	__xmm@00000000000000010000000000000000
	.section	.rdata,"dr",discard,__xmm@00000000000000010000000000000000
	.p2align	4, 0x0
__xmm@00000000000000010000000000000000:
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	1
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.byte	0
	.text
	.globl	PHOTO_mp_VEC_RANGING
	.p2align	4, 0x90
PHOTO_mp_VEC_RANGING:
.seh_proc PHOTO_mp_VEC_RANGING
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$160, %rsp
	.seh_stackalloc 160
	movdqa	%xmm15, 144(%rsp)
	.seh_savexmm %xmm15, 144
	movdqa	%xmm14, 128(%rsp)
	.seh_savexmm %xmm14, 128
	movaps	%xmm13, 112(%rsp)
	.seh_savexmm %xmm13, 112
	movaps	%xmm12, 96(%rsp)
	.seh_savexmm %xmm12, 96
	movaps	%xmm11, 80(%rsp)
	.seh_savexmm %xmm11, 80
	movdqa	%xmm10, 64(%rsp)
	.seh_savexmm %xmm10, 64
	movdqa	%xmm9, 48(%rsp)
	.seh_savexmm %xmm9, 48
	movaps	%xmm8, 32(%rsp)
	.seh_savexmm %xmm8, 32
	movdqa	%xmm7, 16(%rsp)
	.seh_savexmm %xmm7, 16
	movaps	%xmm6, (%rsp)
	.seh_savexmm %xmm6, 0
	.seh_endprologue
	movq	(%rcx), %rax
	movq	48(%rcx), %r10
	movq	56(%rcx), %rcx
	testq	%r10, %r10
	jle	.LBB27_1
	cmpq	$8, %r10
	jae	.LBB27_9
	movss	__real@7f800000(%rip), %xmm2
	movss	__real@ff800000(%rip), %xmm1
	jmp	.LBB27_11
.LBB27_1:
	movss	__real@ff7fffff(%rip), %xmm4
	movss	__real@7f7fffff(%rip), %xmm0
	jmp	.LBB27_2
.LBB27_9:
	movq	%r10, %r11
	shrq	$3, %r11
	leaq	(,%rcx,8), %rsi
	movss	__real@7f800000(%rip), %xmm2
	movss	__real@ff800000(%rip), %xmm1
	movq	%rax, %rdi
	.p2align	4, 0x90
.LBB27_10:
	movss	(%rdi), %xmm0
	movaps	%xmm0, %xmm3
	minss	%xmm2, %xmm3
	maxss	%xmm1, %xmm0
	leaq	(%rdi,%rcx), %rbx
	movss	(%rdi,%rcx), %xmm1
	movaps	%xmm1, %xmm2
	minss	%xmm3, %xmm2
	maxss	%xmm0, %xmm1
	movss	(%rcx,%rbx), %xmm0
	addq	%rcx, %rbx
	movaps	%xmm0, %xmm3
	minss	%xmm2, %xmm3
	maxss	%xmm1, %xmm0
	movss	(%rcx,%rbx), %xmm1
	addq	%rcx, %rbx
	movaps	%xmm1, %xmm2
	minss	%xmm3, %xmm2
	maxss	%xmm0, %xmm1
	movss	(%rcx,%rbx), %xmm0
	addq	%rcx, %rbx
	movaps	%xmm0, %xmm3
	minss	%xmm2, %xmm3
	maxss	%xmm1, %xmm0
	movss	(%rcx,%rbx), %xmm1
	addq	%rcx, %rbx
	movaps	%xmm1, %xmm2
	minss	%xmm3, %xmm2
	maxss	%xmm0, %xmm1
	movss	(%rcx,%rbx), %xmm0
	addq	%rcx, %rbx
	movaps	%xmm0, %xmm3
	minss	%xmm2, %xmm3
	maxss	%xmm1, %xmm0
	movss	(%rcx,%rbx), %xmm1
	movaps	%xmm1, %xmm2
	minss	%xmm3, %xmm2
	maxss	%xmm0, %xmm1
	addq	%rsi, %rdi
	decq	%r11
	jne	.LBB27_10
.LBB27_11:
	movq	%r10, %rsi
	andq	$-8, %rsi
	movq	%r10, %r11
	subq	%rsi, %r11
	jne	.LBB27_13
	movaps	%xmm2, %xmm0
	movaps	%xmm1, %xmm4
	jmp	.LBB27_2
.LBB27_13:
	movq	%r10, %rsi
	shrq	$3, %rsi
	imulq	%rcx, %rsi
	leaq	(%rax,%rsi,8), %rsi
	.p2align	4, 0x90
.LBB27_14:
	movss	(%rsi), %xmm4
	movaps	%xmm4, %xmm0
	minss	%xmm2, %xmm0
	maxss	%xmm1, %xmm4
	addq	%rcx, %rsi
	movaps	%xmm4, %xmm1
	movaps	%xmm0, %xmm2
	decq	%r11
	jne	.LBB27_14
.LBB27_2:
	xorl	%r11d, %r11d
	testq	%r10, %r10
	cmovgq	%r10, %r11
	testl	%r11d, %r11d
	jle	.LBB27_18
	movss	(%r8), %xmm1
	movss	(%rdx), %xmm2
	subss	%xmm2, %xmm1
	subss	%xmm0, %xmm4
	movss	__real@3f800000(%rip), %xmm3
	divss	%xmm4, %xmm3
	incl	%r11d
	movslq	%r11d, %rdx
	decq	%rdx
	movq	%rdx, %r8
	andq	$-4, %r8
	je	.LBB27_4
	movq	%rax, %xmm4
	pshufd	$68, %xmm4, %xmm4
	movaps	%xmm0, %xmm5
	shufps	$0, %xmm0, %xmm5
	movaps	%xmm2, %xmm6
	shufps	$0, %xmm2, %xmm6
	movq	%rcx, %xmm7
	pshufd	$68, %xmm7, %xmm7
	movaps	%xmm3, %xmm8
	mulss	%xmm1, %xmm8
	shufps	$0, %xmm8, %xmm8
	xorl	%r10d, %r10d
	movdqa	__xmm@00000000000000030000000000000002(%rip), %xmm9
	movdqa	__xmm@00000000000000010000000000000000(%rip), %xmm10
	.p2align	4, 0x90
.LBB27_16:
	movq	%r10, %xmm11
	pshufd	$68, %xmm11, %xmm11
	movdqa	%xmm11, %xmm12
	por	%xmm9, %xmm12
	por	%xmm10, %xmm11
	movdqa	%xmm7, %xmm13
	psrlq	$32, %xmm13
	movdqa	%xmm13, %xmm14
	pmuludq	%xmm11, %xmm14
	movdqa	%xmm7, %xmm15
	pmuludq	%xmm11, %xmm15
	psrlq	$32, %xmm11
	pmuludq	%xmm7, %xmm11
	paddq	%xmm14, %xmm11
	psllq	$32, %xmm11
	pmuludq	%xmm12, %xmm13
	movdqa	%xmm7, %xmm14
	pmuludq	%xmm12, %xmm14
	psrlq	$32, %xmm12
	pmuludq	%xmm7, %xmm12
	paddq	%xmm13, %xmm12
	psllq	$32, %xmm12
	paddq	%xmm4, %xmm14
	paddq	%xmm12, %xmm14
	paddq	%xmm4, %xmm15
	paddq	%xmm11, %xmm15
	movq	%xmm15, %r11
	pshufd	$238, %xmm15, %xmm11
	movq	%xmm11, %rsi
	movq	%xmm14, %rdi
	pshufd	$238, %xmm14, %xmm11
	movq	%xmm11, %rbx
	movss	(%rdi), %xmm11
	movss	(%rbx), %xmm12
	unpcklps	%xmm12, %xmm11
	movss	(%r11), %xmm12
	movss	(%rsi), %xmm13
	unpcklps	%xmm13, %xmm12
	movlhps	%xmm11, %xmm12
	subps	%xmm5, %xmm12
	mulps	%xmm8, %xmm12
	addps	%xmm6, %xmm12
	movups	%xmm12, (%r9,%r10,4)
	addq	$4, %r10
	cmpq	%r8, %r10
	jl	.LBB27_16
	cmpq	%r8, %rdx
	jne	.LBB27_5
	jmp	.LBB27_18
.LBB27_4:
	xorl	%r8d, %r8d
.LBB27_5:
	movq	%r8, %r10
	imulq	%rcx, %r10
	addq	%r10, %rax
	.p2align	4, 0x90
.LBB27_6:
	movss	(%rax), %xmm4
	subss	%xmm0, %xmm4
	mulss	%xmm1, %xmm4
	mulss	%xmm3, %xmm4
	addss	%xmm2, %xmm4
	movss	%xmm4, (%r9,%r8,4)
	incq	%r8
	addq	%rcx, %rax
	cmpq	%r8, %rdx
	jne	.LBB27_6
.LBB27_18:
	movaps	(%rsp), %xmm6
	movaps	16(%rsp), %xmm7
	movaps	32(%rsp), %xmm8
	movaps	48(%rsp), %xmm9
	movaps	64(%rsp), %xmm10
	movaps	80(%rsp), %xmm11
	movaps	96(%rsp), %xmm12
	movaps	112(%rsp), %xmm13
	movaps	128(%rsp), %xmm14
	movaps	144(%rsp), %xmm15
	addq	$160, %rsp
	popq	%rbx
	popq	%rdi
	popq	%rsi
	retq
	.seh_endproc

	.lcomm	PHOTO_mp_SPINUP2$TFROOT,3996,32
	.lcomm	PHOTO_mp_SPINUP2$TAWOOD,3996,32
	.lcomm	PHOTO_mp_SPINUP2$TLEAF,3996,32
	.lcomm	PHOTO_mp_SPINUP2$AFROOT,3996,32
	.lcomm	PHOTO_mp_SPINUP2$AAWOOD,3996,32
	.lcomm	PHOTO_mp_SPINUP2$ALEAF,3996,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$MAX_INDEX,4,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$TOTAL_W_PFT,7992,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$TOTAL_BIOMASS_PFT,7992,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$CAWOOD,7992,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$IS_LIVING,1998,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$CFROOT,7992,32
	.lcomm	PHOTO_mp_PFT_AREA_FRAC$CLEAF,7992,32
	.section	.drectve,"yni"
	.ascii	" /DEFAULTLIB:libcmt"
	.ascii	" /DEFAULTLIB:ifconsol.lib"
	.ascii	" /DEFAULTLIB:libifcoremt.lib"
	.ascii	" /DEFAULTLIB:libifport.lib"
	.ascii	" /DEFAULTLIB:libircmt"
	.ascii	" /DEFAULTLIB:libmmt"
	.ascii	" /DEFAULTLIB:oldnames"
	.ascii	" /DEFAULTLIB:svml_dispmt"
	.globl	_fltused
