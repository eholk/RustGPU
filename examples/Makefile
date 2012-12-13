
.phony: all
all: add-float


%.ptx: %.rs
	$(RUSTC) -Zptx -S -O --lib $<

%: %.rs %-kernel.ptx
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<
