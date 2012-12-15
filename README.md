B649 Final Project: Compiling Rust for GPUs
==========

This repository contains a proof of concept for writing GPU kernels in Rust. The NVPTX LLVM backend is used to translate Rust code into PTX code for execution on the GPU. OpenCL is used to handle memory allocation, data transfer and kernel invocation.

Getting Started
----------

This repository includes a custom version of Rust as a submodule. You will need to build this first. Do so with the following command:

    make rustc

You will probably want to go get a cup of tea. The build will take about 30 minutes.

Once that's done, you can build the Rust OpenCL bindings and several example programs:

    make

This will create several executables in the `examples` directory that illustrate various features of our proof of concept.

About the examples
----------

We current have four examples:

1. `add-float` demonstrates adding a single floating point number on the GPU. This is about the simplest possible kernel of any interest.
2. `thread-id` demonstrates the use of Rust intrinsics to determine the current thread id. Without vector indexing, however, there is still only one thread with observable effects.
3. `enum` demonstrates the use of Rust `enum` types. This shows that Rust on the GPU can handle more than just simple scalar data types.
4. `add-vector` shows how to compute on multiple elements of an array. This uses unsafe pointers and raw pointer arithmetic. Furthermore, it currently requires manually modification of the generated PTX because Rust does not yet generate the correct address space for unsafe pointer types. This example primarily shows it is possible to process multiple data elements, although clearly the language support needs to be greatly improved.
