#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdio.h>

void handle_error(cl_int e);
void check_status(cl_int e);
void print_vector(float *x, int len);

cl_platform_id g_platform;
cl_device_id g_device;
cl_context g_context;
cl_program g_program;
cl_command_queue g_queue;

// z = y + y
const char *g_prog_src = 
    "__kernel void add_vectors(__global __read_only float *x,"
    "                          __global __read_only float *y,"
    "                          __global __write_only float *z)"
    "{"
    "    int i = get_global_id(0);"
    "    z[i] = x[i] + y[i];"
    "}";

int main() {
    cl_int status;

    // Find the platforms
    cl_platform_id *platforms = NULL;
    cl_uint nPlatforms = 0;

    // Call once with NULL to determine how much space we need to
    // allocate.
    status = clGetPlatformIDs(0, NULL, &nPlatforms);
    check_status(status);

    printf("Found %d platforms.\n", nPlatforms);

    // Allocate space for the platform IDs.
    platforms = calloc(nPlatforms, sizeof(cl_platform_id));

    // Get the platform IDs.
    status = clGetPlatformIDs(nPlatforms, platforms, &nPlatforms);
    check_status(status);

    // Pick the first platform.
    g_platform = platforms[0];

    // Find out how many devices there are.
    cl_uint n_dev = 0;
    cl_device_type type =
        CL_DEVICE_TYPE_GPU |
        CL_DEVICE_TYPE_CPU |
        CL_DEVICE_TYPE_ACCELERATOR;
    status = clGetDeviceIDs(g_platform, type, CL_UINT_MAX, NULL, &n_dev);
    check_status(status);

    printf("Found %d devices.\n", n_dev);

    // Allocate space for the device IDs
    cl_device_id *devices = NULL;
    devices = calloc(n_dev, sizeof(cl_device_id));

    // Get the device IDs
    status = clGetDeviceIDs(g_platform, type, CL_UINT_MAX,
                            devices, &n_dev);
    check_status(status);

    // Arbitrarily pick the first device.
    g_device = devices[0];

    // Create a context for the devices.
    g_context = clCreateContext(0, n_dev, devices,
                                NULL, // This could be a pointer to a
                                      // notify function.

                                NULL, // And this could be a pointer
                                      // to some application specific
                                      // data that is passed to the
                                      // notify function.
                                &status);
    check_status(status);

    printf("Created context.\n");

    // Create a program
    g_program = clCreateProgramWithSource(g_context,
                                          1, // The source is made up
                                             // of one string.
                                          &g_prog_src,
                                          NULL, // All one of the
                                                // strings are
                                                // NULL-terminated.
                                          &status);
    check_status(status);
    
    printf("Created program.\n");

    status = clBuildProgram(g_program,
                            0, // build for all devices
                            NULL, // the device list is null, since
                                  // we're building for all devices.
                            NULL, // no compiler options
                            NULL, // no notify function
                            NULL);// no notify data
    check_status(status);
    
    printf("Built program.\n");

    // Now we'll set up some vectors to get ready for the kernel.
    float x[] = { 1, 2, 3, 4 };
    float y[] = { 5, 6, 7, 8 };
    float z[sizeof(x) / sizeof(x[0])];

    // And create memory buffers.
    cl_mem clx = clCreateBuffer(g_context,
                                CL_MEM_READ_ONLY,
                                sizeof(x),
                                NULL, // host pointer...
                                &status);
    check_status(status);

    cl_mem cly = clCreateBuffer(g_context,
                                CL_MEM_READ_ONLY,
                                sizeof(y),
                                NULL, // host pointer...
                                &status);
    check_status(status);

    cl_mem clz = clCreateBuffer(g_context,
                                CL_MEM_WRITE_ONLY,
                                sizeof(z),
                                NULL, // host pointer...
                                &status);
    check_status(status);

    printf("Created buffers.\n");

    // Create the kernel
    cl_kernel kernel = clCreateKernel(g_program,
                                      "add_vectors",
                                      &status);
    check_status(status);

    printf("Created kernel.\n");

	// Create the command queue
	g_queue =
	  clCreateCommandQueue(g_context,
						   g_device,
						   0, // flags, such as
						   // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLED
						   &status);
	check_status(status);

	printf("Created command queue.\n");

	// Copy the data
	status = clEnqueueWriteBuffer(g_queue,
								  clx,
								  CL_TRUE, // blocking
								  0, // offset
								  sizeof(x), // how many bytes to copy
								  x,
								  0, // no events in wait list
								  NULL, // no wait list
								  NULL); // We'll ignore the returned event.
	check_status(status);
	printf("Copied x.\n");

	status = clEnqueueWriteBuffer(g_queue,
								  cly,
								  CL_TRUE, // blocking
								  0, // offset
								  sizeof(y), // how many bytes to copy
								  y,
								  0, // no events in wait list
								  NULL, // no wait list
								  NULL); // We'll ignore the returned event.
	check_status(status);
	printf("Copied y.\n");

	// Set the kernel arguments
	// x
	status = clSetKernelArg(kernel,
							0, // x is index 0
							sizeof(clx), // size of the argument
							&clx); // use x's cl_mem object, instead of
	                              // x directly.
	check_status(status);
	printf("Set x\n");

	// y
	status = clSetKernelArg(kernel,
							1, // y is index 1
							sizeof(cly), // size of the argument
							&cly); // use y's cl_mem object, instead of
	                              // y directly.
	check_status(status);
	printf("Set y\n");

	status = clSetKernelArg(kernel,
							2, // z is index 2
							sizeof(clz), // size of the argument
							&clz); // use z's cl_mem object, instead of
	                              // z directly.
	check_status(status);
	printf("Set z\n");

	// Do to kernel
	cl_event event;
	size_t global_size = sizeof(x) / sizeof(x[0]); // number of elements in x
	size_t local_size = 1;
	status = clEnqueueNDRangeKernel(g_queue,
									kernel,
									1, // one dimensional
									NULL, // must be NULL
									&global_size,
									&local_size,
									0, // wait on no events
									NULL, // no event wait list
									&event); // the event we use to
	                                         // tell when this kernel is done.
	check_status(status);
	printf("Executing kernel...\n");

	// Wait for the kernel to complete
	status = clWaitForEvents(1, // one event
							 &event);
	check_status(status);
	clReleaseEvent(event);
	
	printf("Completed kernel.\n");

	// read the results
	status = clEnqueueReadBuffer(g_queue,
								 clz,
								 CL_TRUE, // blocking
								 0, // offset
								 sizeof(z), // how many bytes to read
								 z, // where to store the bytes
								 0, // wait for no events
								 NULL, // no events to wait on
								 NULL); // no event, this is a blocking call.
	check_status(status);

	printf("Read results.\n");

	printf("x: ");
	print_vector(x, sizeof(x) / sizeof(x[0]));
	printf("y: ");
	print_vector(y, sizeof(y) / sizeof(y[0]));
	printf("z: ");
	print_vector(z, sizeof(z) / sizeof(z[0]));

	// Clean up
	clReleaseCommandQueue(g_queue);
    clReleaseProgram(g_program);
    clReleaseContext(g_context);
	clReleaseKernel(kernel);
	clReleaseMemObject(clx);
	clReleaseMemObject(cly);
	clReleaseMemObject(clz);
    free(platforms);
    free(devices);
    return 0;
}

void print_vector(float *x, int len) {
  printf("[");
  for(int i = 0; i < len; ++i) {
	printf(" %f", x[i]);
  }
  printf(" ]\n");
}

void check_status(cl_int e) {
    if(CL_SUCCESS != e) {
        handle_error(e);
    }
}

void handle_error(cl_int e) {
#define HANDLE(x)                                                       \
    if(e == x) {                                                        \
        fprintf(stderr, "OpenCL Error: " #x " (%d)\n", e);              \
		abort();														\
    }
 
    HANDLE(CL_BUILD_PROGRAM_FAILURE);
    HANDLE(CL_COMPILER_NOT_AVAILABLE);
    HANDLE(CL_DEVICE_NOT_FOUND);
	HANDLE(CL_INVALID_ARG_INDEX);
	HANDLE(CL_INVALID_ARG_SIZE);
	HANDLE(CL_INVALID_ARG_VALUE);
    HANDLE(CL_INVALID_BINARY);
    HANDLE(CL_INVALID_BUILD_OPTIONS);
    HANDLE(CL_INVALID_COMMAND_QUEUE);
    HANDLE(CL_INVALID_CONTEXT);
    HANDLE(CL_INVALID_DEVICE);
    HANDLE(CL_INVALID_DEVICE_TYPE);
    HANDLE(CL_INVALID_EVENT_WAIT_LIST);
    HANDLE(CL_INVALID_GLOBAL_OFFSET);
    //HANDLE(CL_INVALID_GLOBAL_WORK_SIZE);
    HANDLE(CL_INVALID_IMAGE_SIZE);
    HANDLE(CL_INVALID_KERNEL);
    HANDLE(CL_INVALID_KERNEL_ARGS);
	HANDLE(CL_INVALID_MEM_OBJECT);
    HANDLE(CL_INVALID_OPERATION);
    HANDLE(CL_INVALID_PLATFORM);
    HANDLE(CL_INVALID_PROGRAM);
    HANDLE(CL_INVALID_PROGRAM_EXECUTABLE);
    HANDLE(CL_INVALID_QUEUE_PROPERTIES);
	HANDLE(CL_INVALID_SAMPLER);
    HANDLE(CL_INVALID_VALUE);
    HANDLE(CL_INVALID_WORK_DIMENSION);
    HANDLE(CL_INVALID_WORK_GROUP_SIZE);
    HANDLE(CL_INVALID_WORK_ITEM_SIZE);
    HANDLE(CL_MEM_OBJECT_ALLOCATION_FAILURE);
    //HANDLE(CL_MISALIGNED_SUB_BUFFER_OFFSET);
    HANDLE(CL_OUT_OF_RESOURCES);
    HANDLE(CL_OUT_OF_HOST_MEMORY);

    fprintf(stderr, "Unknown OpenCL Error: %d\n", e);
    abort();
}
