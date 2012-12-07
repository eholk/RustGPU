HOST = $(shell llvm-config --host-target)

ROOT = $(dir $(realpath $(lastword $(MAKEFILE_LIST))))

RUSTC = $(ROOT)rust/build/$(HOST)/stage2/bin/rustc

.PHONY: submodules all libOpenCL rustc

all: submodules libOpenCL

submodules: .gitmodules
	git submodule update --init

rustc: submodules
	mkdir -p rust/build && \
	cd rust/build && \
	../configure --disable-valgrind && \
	make -j8 && \
	cd ../../

libOpenCL: submodules
	make -C rust-opencl RUSTC=$(RUSTC)
