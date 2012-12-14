
.phony: all
all: add-float thread-id add-vector

.SECONDARY:

%.ptx: %.rs
	$(RUSTC) -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<
