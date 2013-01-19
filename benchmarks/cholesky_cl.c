#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <linux/time.h>
#include <time.h>
#include <inttypes.h>

#include "cholesky_cl.h"

#define round_up(s, m) (((s + m - 1) / m) * m)

void handle_error(cl_int e);
void check_status(cl_int e);
void print_vector(double *x, int len);

cl_platform_id g_platform;
cl_device_id g_device;
cl_context g_context;
cl_program g_program;
cl_command_queue g_queue;

uint64_t nanotime() {
#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    return time_nano;  
#else
    uint64_t ns_per_s = 1000000000LL;
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (ts.tv_sec * ns_per_s + ts.tv_nsec);
#endif    
}

double time_s() {
    return nanotime() / 1e9;
}

void cholesky_init() {
    cl_int status;

   // Find the platforms
    cl_platform_id *platforms = NULL;
    cl_uint nPlatforms = 0;

    // Call once with NULL to determine how much space we need to
    // allocate.
    status = clGetPlatformIDs(0, NULL, &nPlatforms);
    check_status(status);

    //printf("Found %d platforms.\n", nPlatforms);

    // Allocate space for the platform IDs.
    platforms = calloc(nPlatforms, sizeof(cl_platform_id));

    // Get the platform IDs.
    status = clGetPlatformIDs(nPlatforms, platforms, &nPlatforms);
    check_status(status);

    // Find a device. This may involve checking multiple platforms.
    cl_uint n_dev = 0;
    cl_device_id *devices = NULL;
    for(int i = 0; i < nPlatforms; ++i) {
        // Pick the first platform.
        g_platform = platforms[i];

        // Find out how many devices there are.
        cl_device_type type =
            CL_DEVICE_TYPE_GPU |
            CL_DEVICE_TYPE_ACCELERATOR;
        status = clGetDeviceIDs(g_platform, type,
                                CL_UINT_MAX, NULL, &n_dev);
        if(status == CL_DEVICE_NOT_FOUND) {
            continue;
        }
        check_status(status);

        //printf("Found %d devices.\n", n_dev);

        // Allocate space for the device IDs
        devices = calloc(n_dev, sizeof(cl_device_id));

        // Get the device IDs
        status = clGetDeviceIDs(g_platform, type, CL_UINT_MAX,
                                devices, &n_dev);
        check_status(status);

        // Arbitrarily pick the first device.
        g_device = devices[0];
    }

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

    //printf("Created context.\n");

    // Create a program
    size_t prog_size = cholesky_cl_len;
    const char *prog_src = &cholesky_cl[0];
    g_program = clCreateProgramWithSource(g_context,
                                          1, // The source is made up
                                             // of one string.
                                          &prog_src,
                                          &prog_size,
                                          &status);
    check_status(status);
    
    //printf("Created program.\n");

    status = clBuildProgram(g_program,
                            0, // build for all devices
                            NULL, // the device list is null, since
                                  // we're building for all devices.
                            NULL, // no compiler options
                            NULL, // no notify function
                            NULL);// no notify data
    check_status(status);

    // Create the command queue
    g_queue =
      clCreateCommandQueue(g_context,
			   g_device,
			   0, // flags, such as
			   // CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLED
			   &status);
    check_status(status);    
}

void cholesky(uint64_t N, double *data) {
    cl_int status;

    int byte_size = N * N * sizeof(double);

    cl_kernel update_kk = clCreateKernel(g_program,
                                         "update_kk",
                                         &status);
    check_status(status);

    cl_kernel update_k = clCreateKernel(g_program,
                                         "update_k",
                                         &status);
    check_status(status);

    cl_kernel update_block = clCreateKernel(g_program,
                                             "update_block",
                                             &status);
    check_status(status);

    // Create buffers and copy data

    double start = time_s();

    cl_mem cldata = clCreateBuffer(g_context,
                                   CL_MEM_READ_ONLY,
                                   byte_size,
                                   NULL, // host pointer...
                                   &status);
    check_status(status);

    status = clEnqueueWriteBuffer(g_queue,
                                  cldata,
                                  CL_TRUE, // blocking
                                  0, // offset
                                  byte_size, // how many bytes to copy
                                  data,
                                  0, // no events in wait list
                                  NULL, // no wait list
                                  NULL); // We'll ignore the returned event.
    check_status(status);

    // Do the computation

    double compute_start = time_s();

    for(uint64_t k = 0; k < N; ++k) {
        // update_kk
        {
            //printf("update_kk\n");
            status = clSetKernelArg(update_kk,
                                    0,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_kk,
                                1,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_kk,
                                    2,
                                    sizeof(N),
                                    &N);
            check_status(status);
        
            status = clSetKernelArg(update_kk,
                                    3,
                                    sizeof(cl_mem),
                                    &cldata);
            check_status(status);
            
            status = clSetKernelArg(update_kk,
                                    4,
                                    sizeof(k),
                                    &k);
            check_status(status);

            cl_event event;
            size_t local_size = 1;
            size_t global_size = 1;
            status = clEnqueueNDRangeKernel(g_queue,
                                            update_kk,
                                            1, // one dimensional
                                            NULL, // must be NULL
                                            &global_size,
                                            &local_size,
                                            0, // wait on no events
                                            NULL, // no event wait list
                                            &event); // the event we use to
            // tell when this kernel is done.
            check_status(status);

            status = clWaitForEvents(1, // one event
                                     &event);
            check_status(status);
            clReleaseEvent(event);

        }

        // update_k
        {
            //printf("update_k\n");
            status = clSetKernelArg(update_k,
                                    0,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                1,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                    2,
                                    sizeof(N),
                                    &N);
            check_status(status);
        
            status = clSetKernelArg(update_k,
                                    3,
                                    sizeof(cl_mem),
                                    &cldata);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                    4,
                                    sizeof(k),
                                    &k);
            check_status(status);

            cl_event event;
            size_t local_size = 1024;
            size_t global_size = round_up(N, local_size);
            status = clEnqueueNDRangeKernel(g_queue,
                                            update_k,
                                            1, // one dimensional
                                            NULL, // must be NULL
                                            &global_size,
                                            &local_size,
                                            0, // wait on no events
                                            NULL, // no event wait list
                                            &event);
            check_status(status);

            status = clWaitForEvents(1, // one event
                                     &event);
            check_status(status);
            clReleaseEvent(event);
            
        }

        // update_block
        {
            //printf("update_k\n");
            status = clSetKernelArg(update_k,
                                    0,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                1,
                                    sizeof(cl_mem),
                                    NULL);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                    2,
                                    sizeof(N),
                                    &N);
            check_status(status);
        
            status = clSetKernelArg(update_k,
                                    3,
                                    sizeof(cl_mem),
                                    &cldata);
            check_status(status);
            
            status = clSetKernelArg(update_k,
                                    4,
                                    sizeof(k),
                                    &k);
            check_status(status);

            cl_event event;
            size_t local_size[] = {32, 32};
            size_t global_size[] = {round_up(N, local_size[0]),
                                  round_up(N, local_size[1])};
            status = clEnqueueNDRangeKernel(g_queue,
                                            update_k,
                                            2, // one dimensional
                                            NULL, // must be NULL
                                            global_size,
                                            local_size,
                                            0, // wait on no events
                                            NULL, // no event wait list
                                            &event); // the event we use to
            // tell when this kernel is done.
            check_status(status);

            status = clWaitForEvents(1, // one event
                                     &event);
            check_status(status);
            clReleaseEvent(event);
            
        }
    }

    double compute_stop = time_s();

    // Read the results
    status = clEnqueueReadBuffer(g_queue,
                                 cldata,
                                 CL_TRUE,
                                 0,
                                 byte_size,
                                 data,
                                 0,
                                 NULL,
                                 NULL);
    check_status(status);

    double stop = time_s();

    // Print the results
    printf("%lf\t%lf\n", compute_stop - compute_start, stop - start);
}

void print_vector(double *x, int len) {
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
    HANDLE(CL_INVALID_KERNEL_NAME);
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
