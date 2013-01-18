#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
 
// Enable double precision values
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                      "\n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
  "\n" ;


void handle_error(cl_int e) {
#define HANDLE(x)                                                       \
  if(e == x) {                                                        \
    fprintf(stderr, "OpenCL Error: " #x " (%d)\n", e);              \
    abort();\
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

void check_status(cl_int e) {
  if(CL_SUCCESS != e) {
    handle_error(e);
  }
}

int main( int argc, char* argv[] )
{
  // Length of vectors
  unsigned int n = 100000;
 
  // Host input vectors
  double *h_a;
  double *h_b;
  // Host output vector
  double *h_c;
 
  // Device input buffers
  cl_mem d_a;
  cl_mem d_b;
  // Device output buffer

  cl_mem d_c;
 
  cl_platform_id cpPlatform;        // OpenCL platform
  cl_device_id device_id;           // device ID
  cl_context context;               // context
  cl_command_queue queue;           // command queue
  cl_program program;               // program
  cl_kernel kernel;                 // kernel
  cl_uint num_platforms;
 
  // Size, in bytes, of each vector
  size_t bytes = n*sizeof(double);
 
  // Allocate memory for each vector on host
  h_a = (double*)malloc(bytes);
  h_b = (double*)malloc(bytes);
  h_c = (double*)malloc(bytes);
 
  // Initialize vectors on host
  int i;
  for( i = 0; i < n; i++ )
    {
      h_a[i] = sinf(i)*sinf(i);
      h_b[i] = cosf(i)*cosf(i);
    }
 
  size_t globalSize, localSize;
  cl_int err;
 
  // Number of work items in each local work group
  localSize = 64;
 
  // Number of total work items - localSize must be devisor
  globalSize = ceil(n/(float)localSize)*localSize;
 
  // Bind to platform
  err = clGetPlatformIDs(0, NULL, &num_platforms);
  check_status(err);

  for(i = 0; i < num_platforms; i++){
  }

  // Get ID for the device
  err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  check_status(err);

  // Create a context  
  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
 
  // Create a command queue 
  queue = clCreateCommandQueue(context, device_id, 0, &err);
 
  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1,
				      (const char **) & kernelSource, NULL, &err);
 
  // Build the program executable 
  clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
  // Create the compute kernel in the program we wish to run
  kernel = clCreateKernel(program, "vecAdd", &err);
 
  // Create the input and output arrays in device memory for our calculation
  d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
  d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
  // Write our data set into the input array in device memory
  err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
			     bytes, h_a, 0, NULL, NULL);
  err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
			      bytes, h_b, 0, NULL, NULL);
 
  // Set the arguments to our compute kernel
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
 
  // Execute the kernel over the entire range of the data set  
  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
			       0, NULL, NULL);
 
  // Wait for the command queue to get serviced before reading back results
  clFinish(queue);
 
  // Read the results from the device
  clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
		      bytes, h_c, 0, NULL, NULL );
 
  //Sum up vector c and print result divided by n, this should equal 1 within error
  double sum = 0;
  for(i=0; i<n; i++)
    sum += h_c[i];
  printf("final result: %f\n", sum/n);
 
  // release OpenCL resources
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
 
  //release host memory
  free(h_a);
  free(h_b);
  free(h_c);
 
  return 0;
}
