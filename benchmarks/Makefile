
EXAMPLES = add-vector mat-mult cholesky

ifndef RUSTC
	RUSTC = rustc
endif

ifndef ROOT
	ROOT = ../
endif

CFLAGS = -std=c99 -g -O3

.phony: all
all: $(EXAMPLES) cholesky-rustcl cholesky-opencl

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

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
