	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"global.f90"
	.def	TYPES.;
	.scl	2;
	.type	32;
	.endef
	.globl	TYPES.
	.p2align	4, 0x90
TYPES.:
	retq

	.def	GLOBAL_PAR.;
	.scl	2;
	.type	32;
	.endef
	.globl	GLOBAL_PAR.
	.p2align	4, 0x90
GLOBAL_PAR.:
	retq

	.def	PHOTO_PAR.;
	.scl	2;
	.type	32;
	.endef
	.globl	PHOTO_PAR.
	.p2align	4, 0x90
PHOTO_PAR.:
	retq

	.section	.drectve,"yni"
	.ascii	" /DEFAULTLIB:libcmt"
	.ascii	" /DEFAULTLIB:ifconsol.lib"
	.ascii	" /DEFAULTLIB:libifcoremt.lib"
	.ascii	" /DEFAULTLIB:libifport.lib"
	.ascii	" /DEFAULTLIB:libircmt"
	.ascii	" /DEFAULTLIB:libmmt"
	.ascii	" /DEFAULTLIB:oldnames"
	.ascii	" /DEFAULTLIB:svml_dispmt"
