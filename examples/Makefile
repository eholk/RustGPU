
.phony: all
all: add-float #thread-id add-vector enum

.SECONDARY:

%.ptx: %.rs common.rs
	$(RUSTC) -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<
