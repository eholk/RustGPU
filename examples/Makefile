
EXAMPLES = add-float thread-id add-vector enum map-macro unique-vec \
	vector-sum

# Single examples are files that do not have their kernels in a separate file.
SINGLE_EXAMPLES = add-vector1

.phony: all
all: $(EXAMPLES) $(SINGLE_EXAMPLES)

.SECONDARY:

%.ptx: %.rs common.rs gpu.rs
	$(RUSTC) --cfg gpu -L $(ROOT)/rust-opencl -Zptx -S -O --lib -Zno-asm-comments $<

%: %.rs %-kernel.ptx common.rs
	$(RUSTC) -L $(ROOT)/rust-opencl -O $<

%: %.rs %.ptx common.rs
	$(RUSTC) --cfg no_gpu -L $(ROOT)/rust-opencl -O $<

.phony: clean
clean:
	rm $(EXAMPLES) *.ptx
