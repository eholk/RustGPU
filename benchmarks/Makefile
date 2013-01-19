
EXAMPLES = add-vector mat-mult cholesky jacobi

ifndef RUSTC
	RUSTC = rustc
endif

ifndef ROOT
	ROOT = ../
endif

CFLAGS = -std=c99 -g -O3

.phony: all
all: $(EXAMPLES) cholesky-rustcl cholesky-opencl jacobi-rustcl jacobi-opencl

.SECONDARY:

%.ptx: %.rs common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<

cholesky-rustcl: cholesky-rustcl.rs cholesky.cl
	$(RUSTC) -L $(ROOT)/rust-opencl -O cholesky-rustcl.rs

cholesky-opencl: cholesky-opencl.rs cholesky_cl.o
	$(RUSTC) -L $(ROOT)/rust-opencl -O cholesky-opencl.rs

cholesky_cl.h: cholesky.cl
	xxd -i cholesky.cl > cholesky_cl.h

cholesky_cl.o: cholesky_cl.h cholesky_cl.c

jacobi-rustcl: jacobi-rustcl.rs jacobi.cl
	$(RUSTC) -L $(ROOT)/rust-opencl -O jacobi-rustcl.rs

jacobi-opencl: jacobi-opencl.rs jacobi_cl.o
	$(RUSTC) -L $(ROOT)/rust-opencl -O jacobi-opencl.rs

jacobi_cl.h: jacobi.cl
	xxd -i jacobi.cl > jacobi_cl.h

jacobi_cl.o: jacobi_cl.h jacobi_cl.c

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
