	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"evap.f90"
	.def	WATER.;
	.scl	2;
	.type	32;
	.endef
	.globl	WATER.
	.p2align	4, 0x90
WATER.:
	retq

	.def	WATER_mp_WTT;
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
	.globl	WATER_mp_WTT
	.p2align	4, 0x90
WATER_mp_WTT:
.seh_proc WATER_mp_WTT
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

	.def	WATER_mp_SOIL_TEMP_SUB;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@d73c97c3
	.section	.rdata,"dr",discard,__real@d73c97c3
	.p2align	2, 0x0
__real@d73c97c3:
	.long	0xd73c97c3
	.globl	__real@3f800000
	.section	.rdata,"dr",discard,__real@3f800000
	.p2align	2, 0x0
__real@3f800000:
	.long	0x3f800000
	.globl	__real@3f000000
	.section	.rdata,"dr",discard,__real@3f000000
	.p2align	2, 0x0
__real@3f000000:
	.long	0x3f000000
	.text
	.globl	WATER_mp_SOIL_TEMP_SUB
	.p2align	4, 0x90
WATER_mp_SOIL_TEMP_SUB:
.seh_proc WATER_mp_SOIL_TEMP_SUB
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
	subq	$104, %rsp
	.seh_stackalloc 104
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, %rsi
	movss	__real@d73c97c3(%rip), %xmm0
	callq	expf
	movss	__real@3f800000(%rip), %xmm2
	subss	%xmm0, %xmm2
	movss	WATER_mp_SOIL_TEMP_SUB$T0(%rip), %xmm1
	leaq	4(%rsi), %rax
	movq	%rax, 88(%rsp)
	leaq	8(%rsi), %rax
	movq	%rax, 80(%rsp)
	leaq	16(%rsi), %rax
	movq	%rax, 72(%rsp)
	leaq	20(%rsi), %rax
	movq	%rax, 64(%rsp)
	movq	%rsi, 32(%rsp)
	leaq	24(%rsi), %rax
	movq	%rax, 56(%rsp)
	movl	$1, %eax
	movl	$2, %r8d
	movl	$3, %r9d
	movw	$4, %si
	movl	$5, %r10d
	movl	$6, %r11d
	movl	$7, %ebx
	movw	$8, %r14w
	xorl	%r15d, %r15d
	.p2align	4, 0x90
.LBB2_1:
	movq	%rsi, 96(%rsp)
	movq	%rax, 40(%rsp)
	movq	40(%rsp), %rax
	movabsq	$-6148914691236517205, %rsi
	mulq	%rsi
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	32(%rsp), %rdi
	subq	%rax, %rdi
	movq	%r8, %rax
	mulq	%rsi
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rcx
	movq	88(%rsp), %r12
	movq	%r9, %rax
	mulq	%rsi
	subq	%rcx, %r12
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	80(%rsp), %rcx
	subq	%rax, %rcx
	movq	%r10, %rax
	mulq	%rsi
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	72(%rsp), %rbp
	subq	%rax, %rbp
	movq	%r11, %rax
	mulq	%rsi
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rax
	movq	64(%rsp), %r13
	subq	%rax, %r13
	movq	%rbx, %rax
	mulq	%rsi
	movq	96(%rsp), %rsi
	addq	%rdx, %rdx
	andq	$-16, %rdx
	leaq	(%rdx,%rdx,2), %rdx
	movq	56(%rsp), %rax
	subq	%rdx, %rax
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rdi,%r15,4), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%r12,%r15,4), %xmm1
	movzwl	%si, %edx
	imull	$43691, %edx, %edx
	shrl	$17, %edx
	andl	$-4, %edx
	leal	(%rdx,%rdx,2), %edx
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rcx,%r15,4), %xmm1
	movzwl	%r14w, %ecx
	imull	$43691, %ecx, %ecx
	shrl	$17, %ecx
	andl	$-4, %ecx
	negl	%edx
	addl	%r15d, %edx
	addl	$4, %edx
	testw	%dx, %dx
	movl	$12, %edi
	cmovel	%edi, %edx
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movzwl	%dx, %edx
	movq	32(%rsp), %r12
	mulss	-4(%r12,%rdx,4), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rbp,%r15,4), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%r13,%r15,4), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rax,%r15,4), %xmm1
	leal	(%rcx,%rcx,2), %eax
	negl	%eax
	addl	%r15d, %eax
	addl	$8, %eax
	testw	%ax, %ax
	cmovel	%edi, %eax
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movzwl	%ax, %eax
	mulss	-4(%r12,%rax,4), %xmm1
	movq	40(%rsp), %rax
	addq	$8, %rax
	addq	$8, %r15
	addq	$8, %r8
	addq	$8, %r9
	addl	$8, %esi
	addq	$8, %r10
	addq	$8, %r11
	addq	$8, %rbx
	addl	$8, %r14d
	cmpl	$1088, %r15d
	jne	.LBB2_1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	movq	32(%rsp), %rax
	mulss	32(%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	36(%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	40(%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	44(%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rax), %xmm1
	mulss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	4(%rax), %xmm1
	mulss	%xmm1, %xmm0
	addss	%xmm2, %xmm0
	mulss	8(%rax), %xmm0
	addss	%xmm0, %xmm1
	mulss	__real@3f000000(%rip), %xmm1
	movss	%xmm0, WATER_mp_SOIL_TEMP_SUB$T0(%rip)
	movq	48(%rsp), %rax
	movss	%xmm1, (%rax)
	addq	$104, %rsp
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

	.def	WATER_mp_SOIL_TEMP;
	.scl	2;
	.type	32;
	.endef
	.globl	WATER_mp_SOIL_TEMP
	.p2align	4, 0x90
WATER_mp_SOIL_TEMP:
.seh_proc WATER_mp_SOIL_TEMP
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$48, %rsp
	.seh_stackalloc 48
	movaps	%xmm6, 32(%rsp)
	.seh_savexmm %xmm6, 32
	.seh_endprologue
	movq	%rdx, %rsi
	movss	(%rcx), %xmm6
	movss	__real@d73c97c3(%rip), %xmm0
	callq	expf
	movaps	%xmm6, %xmm2
	mulss	%xmm0, %xmm2
	movss	__real@3f800000(%rip), %xmm1
	subss	%xmm0, %xmm1
	addss	%xmm2, %xmm1
	mulss	(%rsi), %xmm1
	addss	%xmm6, %xmm1
	mulss	__real@3f000000(%rip), %xmm1
	movaps	%xmm1, %xmm0
	movaps	32(%rsp), %xmm6
	addq	$48, %rsp
	popq	%rsi
	retq
	.seh_endproc

	.def	WATER_mp_PENMAN;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@bf800000
	.section	.rdata,"dr",discard,__real@bf800000
	.p2align	2, 0x0
__real@bf800000:
	.long	0xbf800000
	.globl	__real@438c68f643816f5c
	.section	.rdata,"dr",discard,__real@438c68f643816f5c
	.p2align	3, 0x0
__real@438c68f643816f5c:
	.long	0x43816f5c
	.long	0x438c68f6
	.globl	__real@40c39653
	.section	.rdata,"dr",discard,__real@40c39653
	.p2align	2, 0x0
__real@40c39653:
	.long	0x40c39653
	.globl	__real@43806f5c
	.section	.rdata,"dr",discard,__real@43806f5c
	.p2align	2, 0x0
__real@43806f5c:
	.long	0x43806f5c
	.globl	__real@4195d4fe
	.section	.rdata,"dr",discard,__real@4195d4fe
	.p2align	2, 0x0
__real@4195d4fe:
	.long	0x4195d4fe
	.globl	__real@3b900901
	.section	.rdata,"dr",discard,__real@3b900901
	.p2align	2, 0x0
__real@3b900901:
	.long	0x3b900901
	.globl	__real@40c39168
	.section	.rdata,"dr",discard,__real@40c39168
	.p2align	2, 0x0
__real@40c39168:
	.long	0x40c39168
	.globl	__real@438b68f6
	.section	.rdata,"dr",discard,__real@438b68f6
	.p2align	2, 0x0
__real@438b68f6:
	.long	0x438b68f6
	.globl	__real@41b849ba
	.section	.rdata,"dr",discard,__real@41b849ba
	.p2align	2, 0x0
__real@41b849ba:
	.long	0x41b849ba
	.globl	__real@3b446458
	.section	.rdata,"dr",discard,__real@3b446458
	.p2align	2, 0x0
__real@3b446458:
	.long	0x3b446458
	.globl	__real@420f745d
	.section	.rdata,"dr",discard,__real@420f745d
	.p2align	2, 0x0
__real@420f745d:
	.long	0x420f745d
	.globl	__real@459c4000
	.section	.rdata,"dr",discard,__real@459c4000
	.p2align	2, 0x0
__real@459c4000:
	.long	0x459c4000
	.globl	__real@36dd1193
	.section	.rdata,"dr",discard,__real@36dd1193
	.p2align	2, 0x0
__real@36dd1193:
	.long	0x36dd1193
	.globl	__real@3a2cb5bb
	.section	.rdata,"dr",discard,__real@3a2cb5bb
	.p2align	2, 0x0
__real@3a2cb5bb:
	.long	0x3a2cb5bb
	.globl	__real@4140c49c
	.section	.rdata,"dr",discard,__real@4140c49c
	.p2align	2, 0x0
__real@4140c49c:
	.long	0x4140c49c
	.globl	__real@3d10725b
	.section	.rdata,"dr",discard,__real@3d10725b
	.p2align	2, 0x0
__real@3d10725b:
	.long	0x3d10725b
	.text
	.globl	WATER_mp_PENMAN
	.p2align	4, 0x90
WATER_mp_PENMAN:
.seh_proc WATER_mp_PENMAN
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
	movaps	%xmm14, 160(%rsp)
	.seh_savexmm %xmm14, 160
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
	movq	%r9, %rsi
	movq	%r8, %rbx
	movq	%rcx, %rdi
	movss	(%rdx), %xmm9
	movss	__real@3f800000(%rip), %xmm1
	addss	%xmm9, %xmm1
	movss	__real@bf800000(%rip), %xmm10
	addss	%xmm9, %xmm10
	xorps	%xmm8, %xmm8
	xorl	%r14d, %r14d
	ucomiss	%xmm8, %xmm1
	setb	%r14b
	leaq	__real@41b849ba4195d4fe(%rip), %r13
	movss	(%r13,%r14,4), %xmm0
	leaq	__real@3b4464583b900901(%rip), %rbp
	movss	(%rbp,%r14,4), %xmm2
	mulss	%xmm1, %xmm2
	subss	%xmm2, %xmm0
	leaq	__real@438c68f643816f5c(%rip), %rax
	movss	(%rax,%r14,4), %xmm2
	addss	%xmm9, %xmm2
	mulss	%xmm1, %xmm0
	divss	%xmm2, %xmm0
	callq	expf
	movaps	%xmm0, %xmm6
	leaq	__real@40c3916840c39653(%rip), %r12
	ucomiss	%xmm8, %xmm10
	jae	.LBB4_1
	movss	__real@40c39168(%rip), %xmm11
	movss	__real@438b68f6(%rip), %xmm12
	movss	__real@41b849ba(%rip), %xmm7
	movss	__real@3b446458(%rip), %xmm13
	jmp	.LBB4_3
.LBB4_1:
	movss	__real@40c39653(%rip), %xmm11
	movss	__real@43806f5c(%rip), %xmm12
	movss	__real@4195d4fe(%rip), %xmm7
	movss	__real@3b900901(%rip), %xmm13
.LBB4_3:
	xorl	%r15d, %r15d
	ucomiss	%xmm8, %xmm9
	setb	%r15b
	movss	(%r13,%r15,4), %xmm0
	movss	(%rbp,%r15,4), %xmm1
	mulss	%xmm9, %xmm1
	subss	%xmm1, %xmm0
	leaq	__real@438be8f64380ef5c(%rip), %rax
	movss	(%rax,%r15,4), %xmm1
	addss	%xmm9, %xmm1
	mulss	%xmm9, %xmm0
	divss	%xmm1, %xmm0
	callq	expf
	mulss	(%r12,%r15,4), %xmm0
	movss	__real@3f800000(%rip), %xmm8
	subss	(%rbx), %xmm8
	mulss	%xmm0, %xmm8
	movss	__real@420f745d(%rip), %xmm0
	ucomiss	%xmm8, %xmm0
	movq	288(%rsp), %rax
	movss	(%rax), %xmm14
	ja	.LBB4_5
	movss	__real@459c4000(%rip), %xmm0
	ucomiss	%xmm14, %xmm0
	ja	.LBB4_5
	xorps	%xmm0, %xmm0
	jmp	.LBB4_7
.LBB4_5:
	mulss	(%r12,%r14,4), %xmm6
	mulss	%xmm10, %xmm13
	subss	%xmm13, %xmm7
	mulss	%xmm10, %xmm7
	addss	%xmm12, %xmm9
	divss	%xmm9, %xmm7
	movaps	%xmm7, %xmm0
	callq	expf
	mulss	%xmm11, %xmm0
	subss	%xmm0, %xmm6
	mulss	__real@3f000000(%rip), %xmm6
	mulss	__real@36dd1193(%rip), %xmm14
	addss	__real@3a2cb5bb(%rip), %xmm14
	mulss	(%rdi), %xmm14
	movss	(%rsi), %xmm0
	mulss	%xmm6, %xmm0
	mulss	__real@4140c49c(%rip), %xmm8
	addss	%xmm0, %xmm8
	addss	%xmm6, %xmm14
	mulss	__real@3d10725b(%rip), %xmm8
	divss	%xmm14, %xmm8
	xorps	%xmm0, %xmm0
	maxss	%xmm0, %xmm8
	movaps	%xmm8, %xmm0
.LBB4_7:
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

	.def	WATER_mp_AVAILABLE_ENERGY;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@403947ae
	.section	.rdata,"dr",discard,__real@403947ae
	.p2align	2, 0x0
__real@403947ae:
	.long	0x403947ae
	.globl	__real@42514dd3
	.section	.rdata,"dr",discard,__real@42514dd3
	.p2align	2, 0x0
__real@42514dd3:
	.long	0x42514dd3
	.text
	.globl	WATER_mp_AVAILABLE_ENERGY
	.p2align	4, 0x90
WATER_mp_AVAILABLE_ENERGY:
	movss	(%rcx), %xmm0
	mulss	__real@403947ae(%rip), %xmm0
	addss	__real@42514dd3(%rip), %xmm0
	retq

	.def	WATER_mp_RUNOFF;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@40d33333
	.section	.rdata,"dr",discard,__real@40d33333
	.p2align	2, 0x0
__real@40d33333:
	.long	0x40d33333
	.globl	__real@41380000
	.section	.rdata,"dr",discard,__real@41380000
	.p2align	2, 0x0
__real@41380000:
	.long	0x41380000
	.text
	.globl	WATER_mp_RUNOFF
	.p2align	4, 0x90
WATER_mp_RUNOFF:
.seh_proc WATER_mp_RUNOFF
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movss	(%rcx), %xmm0
	movss	__real@40d33333(%rip), %xmm1
	callq	powf
	mulss	__real@41380000(%rip), %xmm0
	addq	$40, %rsp
	retq
	.seh_endproc

	.def	WATER_mp_EVPOT2;
	.scl	2;
	.type	32;
	.endef
	.globl	__real@3aacb5bb
	.section	.rdata,"dr",discard,__real@3aacb5bb
	.p2align	2, 0x0
__real@3aacb5bb:
	.long	0x3aacb5bb
	.globl	__real@00000000
	.section	.rdata,"dr",discard,__real@00000000
	.p2align	2, 0x0
__real@00000000:
	.long	0x00000000
	.text
	.globl	WATER_mp_EVPOT2
	.p2align	4, 0x90
WATER_mp_EVPOT2:
.seh_proc WATER_mp_EVPOT2
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	pushq	%rbx
	.seh_pushreg %rbx
	subq	$208, %rsp
	.seh_stackalloc 208
	movaps	%xmm15, 192(%rsp)
	.seh_savexmm %xmm15, 192
	movaps	%xmm14, 176(%rsp)
	.seh_savexmm %xmm14, 176
	movaps	%xmm13, 160(%rsp)
	.seh_savexmm %xmm13, 160
	movaps	%xmm12, 144(%rsp)
	.seh_savexmm %xmm12, 144
	movaps	%xmm11, 128(%rsp)
	.seh_savexmm %xmm11, 128
	movaps	%xmm10, 112(%rsp)
	.seh_savexmm %xmm10, 112
	movaps	%xmm9, 96(%rsp)
	.seh_savexmm %xmm9, 96
	movaps	%xmm8, 80(%rsp)
	.seh_savexmm %xmm8, 80
	movaps	%xmm7, 64(%rsp)
	.seh_savexmm %xmm7, 64
	movaps	%xmm6, 48(%rsp)
	.seh_savexmm %xmm6, 48
	.seh_endprologue
	movq	%r9, %rsi
	movq	%r8, %rbx
	movq	%rcx, %rdi
	movss	(%rdx), %xmm10
	movss	__real@bf800000(%rip), %xmm14
	addss	%xmm10, %xmm14
	movss	__real@4195d4fe(%rip), %xmm2
	xorps	%xmm0, %xmm0
	ucomiss	%xmm0, %xmm14
	jae	.LBB7_1
	movss	__real@40c39168(%rip), %xmm0
	movss	%xmm0, 44(%rsp)
	movss	__real@438b68f6(%rip), %xmm15
	movss	__real@41b849ba(%rip), %xmm8
	movss	__real@3b446458(%rip), %xmm12
	jmp	.LBB7_3
.LBB7_1:
	movss	__real@40c39653(%rip), %xmm0
	movss	%xmm0, 44(%rsp)
	movss	__real@43806f5c(%rip), %xmm15
	movss	__real@3b900901(%rip), %xmm12
	movaps	%xmm2, %xmm8
.LBB7_3:
	movaps	%xmm10, %xmm1
	addss	__real@3f800000(%rip), %xmm1
	xorl	%eax, %eax
	ucomiss	__real@00000000(%rip), %xmm1
	setb	%cl
	movss	__real@41b849ba(%rip), %xmm6
	movaps	%xmm6, %xmm0
	jb	.LBB7_5
	movaps	%xmm2, %xmm0
.LBB7_5:
	movss	__real@3b446458(%rip), %xmm13
	movss	__real@3b900901(%rip), %xmm3
	movaps	%xmm13, %xmm2
	jb	.LBB7_7
	movaps	%xmm3, %xmm2
.LBB7_7:
	mulss	%xmm1, %xmm2
	subss	%xmm2, %xmm0
	mulss	%xmm1, %xmm0
	movb	%cl, %al
	leaq	__real@438c68f643816f5c(%rip), %rcx
	movss	(%rcx,%rax,4), %xmm1
	addss	%xmm10, %xmm1
	divss	%xmm1, %xmm0
	movss	__real@40c39168(%rip), %xmm11
	movss	__real@40c39653(%rip), %xmm1
	movaps	%xmm11, %xmm9
	jb	.LBB7_9
	movaps	%xmm1, %xmm9
.LBB7_9:
	callq	expf
	movaps	%xmm0, %xmm7
	mulss	%xmm14, %xmm12
	subss	%xmm12, %xmm8
	addss	%xmm10, %xmm15
	mulss	%xmm14, %xmm8
	divss	%xmm15, %xmm8
	movaps	%xmm8, %xmm0
	callq	expf
	movaps	%xmm0, %xmm8
	xorl	%eax, %eax
	ucomiss	__real@00000000(%rip), %xmm10
	setb	%cl
	jb	.LBB7_11
	movss	__real@3b900901(%rip), %xmm13
	movss	__real@4195d4fe(%rip), %xmm6
.LBB7_11:
	mulss	%xmm9, %xmm7
	mulss	44(%rsp), %xmm8
	movb	%cl, %al
	jb	.LBB7_13
	movss	__real@40c39653(%rip), %xmm11
.LBB7_13:
	mulss	%xmm10, %xmm13
	subss	%xmm13, %xmm6
	leaq	__real@438be8f64380ef5c(%rip), %rcx
	mulss	%xmm10, %xmm6
	addss	(%rcx,%rax,4), %xmm10
	divss	%xmm10, %xmm6
	movaps	%xmm6, %xmm0
	callq	expf
	subss	%xmm8, %xmm7
	mulss	__real@3f000000(%rip), %xmm7
	movss	__real@3f800000(%rip), %xmm3
	subss	(%rbx), %xmm3
	mulss	%xmm11, %xmm0
	movss	(%rdi), %xmm1
	mulss	__real@3aacb5bb(%rip), %xmm1
	movss	(%rsi), %xmm2
	mulss	%xmm7, %xmm2
	mulss	__real@4140c49c(%rip), %xmm0
	mulss	%xmm3, %xmm0
	addss	%xmm2, %xmm0
	mulss	__real@3d10725b(%rip), %xmm0
	addss	%xmm7, %xmm1
	divss	%xmm1, %xmm0
	maxss	__real@00000000(%rip), %xmm0
	movaps	48(%rsp), %xmm6
	movaps	64(%rsp), %xmm7
	movaps	80(%rsp), %xmm8
	movaps	96(%rsp), %xmm9
	movaps	112(%rsp), %xmm10
	movaps	128(%rsp), %xmm11
	movaps	144(%rsp), %xmm12
	movaps	160(%rsp), %xmm13
	movaps	176(%rsp), %xmm14
	movaps	192(%rsp), %xmm15
	addq	$208, %rsp
	popq	%rbx
	popq	%rdi
	popq	%rsi
	retq
	.seh_endproc

	.lcomm	WATER_mp_SOIL_TEMP_SUB$T0,4,4
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
