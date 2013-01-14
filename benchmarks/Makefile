
EXAMPLES = add-vector

.phony: all
all: $(EXAMPLES)

.SECONDARY:

%.ptx: %.rs common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
