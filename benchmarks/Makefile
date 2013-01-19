
EXAMPLES = add-vector mat-mult cholesky jacobi

ifndef RUSTC
	RUSTC = rustc
endif

ifndef ROOT
	ROOT = ../
endif

CFLAGS = -std=c99 -g -O3

.phony: all
all: $(EXAMPLES) cholesky-rustcl cholesky-opencl jacobi-rustcl

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

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
