ifndef RUSTC
	RUSTC = rustc
endif

EXAMPLES = add-vector mat-mult

# Single examples are files that do not have their kernels in a separate file.
SINGLE_EXAMPLES = add-vector1

.phony: all
all: $(EXAMPLES)

.SECONDARY:

%.ptx: %.rs common.rs gpu.rs
	$(RUSTC) --cfg gpu -L ../../rust-opencl -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx common.rs
	$(RUSTC) -L ../../rust-opencl -O $<

%: %.rs %.ptx common.rs
	$(RUSTC) --cfg no_gpu -L ../../rust-opencl -O $<

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
