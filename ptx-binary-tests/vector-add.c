#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>

#define DATA_SIZE 1024
// Simple compute kernel which computes the square of an input array 
//
const char *KernelSource = "\n" \
"__kernel void square(                                                       \n" \
"   __global float* input,                                              \n" \
"   __global float* output,                                             \n" \
"   const unsigned int count)                                           \n" \
"{                                                                      \n" \
"   int i = get_global_id(0);                                           \n" \
"   if(i < count)                                                       \n" \
"       output[i] = input[i] * input[i];                                \n" \
"}                                                                      \n" \
"\n";

int main(int argc, char** argv){
  int err;
  float data[DATA_SIZE];
  float results[DATA_SIZE];
  unsigned int correct;

  size_t global;
  size_t local;

  cl_platform_id p_id;
  cl_uint np;
  cl_uint nd;
  cl_device_id device_id;
  cl_context context;
  cl_command_queue commands;
  cl_program program;
  cl_kernel kernel;

  cl_mem input;
  cl_mem output;

  int i = 0;
  unsigned int count = DATA_SIZE;
  for(i = 0; i < count; i++){
    data[i] = rand() / (float)RAND_MAX;
  }

  int gpu = 1;
  clGetPlatformIDs(1, &p_id, &np);
  err = clGetDeviceIDs(p_id, gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, &nd);
  if (err != CL_SUCCESS)
  {
    printf("Error : Failed to create a device group!\n");
    return EXIT_FAILURE;
  }

  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  if (!context)
  {
    printf("Error: Failed to create a compute context!\n");
    return EXIT_FAILURE;
  }

  commands = clCreateCommandQueue(context, device_id, 0, &err);
  if (!commands)
  {
    printf("Error: Failed to create a command commands!\n");
    return EXIT_FAILURE;
  }

  /*
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  if (!program)
  {
    printf("Error: Failed to create compute program!\n");
    return EXIT_FAILURE;
  }

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    size_t len;
    char buffer[2048];
    
    printf("Error: Failed to build program executable!\n");
    clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
    exit(1);
  }
  */
  FILE *fp = fopen("kernel.bin", "rb");
  if (fp == NULL){
    printf("Error: Cannot open kernel binary!\n");
    return EXIT_FAILURE;
  }
  
  fseek(fp, 0, SEEK_END);
  int kernel_length = ftell(fp);
  rewind(fp);
  unsigned char *binary = (unsigned char*)malloc(sizeof(unsigned char) * kernel_length + 10);
  size_t binary_size = fread(binary,1, sizeof(unsigned char) * kernel_length + 10, fp);
  fclose(fp);
  cl_int binary_status, ret;
  program = clCreateProgramWithBinary(context, 1, &device_id, (const size_t *)&binary_size, (const unsigned char **)&binary, &binary_status, &ret);
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);                                                                                                                                                                                 
  if (err != CL_SUCCESS)                                                                                                                                                                                                                    
    {                                                                                                                                                                                                                                         
      size_t len;                                                                                                                                                                                                                             
      char buffer[2048];                                                                                                                                                                                                                      
                                                                                                                                                                                                                                            
      printf("Error: Failed to build program executable!\n");                                                                                                                                                                                 
      clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);                                                                                                                                          
      printf("%s\n", buffer);                                                                                                                                                                                                                 
      exit(1);                                                                                                                                                                                                                                
    }  
  /* size_t kernel_length;
  err= clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &kernel_length, NULL);
  if(err != CL_SUCCESS){
    printf("Error: Failed to get program info!\n");
    return EXIT_FAILURE;
  }

  unsigned char* bin;
  bin = (char*)malloc(sizeof(char)*kernel_length);
  err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, kernel_length, &bin, NULL);
  if(err != CL_SUCCESS){
    printf("Error: Failed to get program binary!\n");
    return EXIT_FAILURE;
  }

  // Print the binary out to the output file
  FILE *fp = fopen("kernel.bin", "wb");    
  fwrite(bin, 1, kernel_length, fp);
  fclose(fp);
  */
  kernel = clCreateKernel(program, "square", &err);
  if (!kernel || err != CL_SUCCESS)
  {
    printf("Error: Failed to create compute kernel!\n");
    exit(1);
  }

  input = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, NULL);
  output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, NULL);
  if (!input || !output)
  {
    printf("Error: Failed to allocate device memory!\n");
    exit(1);
  }

  err = clEnqueueWriteBuffer(commands, input, CL_TRUE, 0, sizeof(float) * count, data, 0, NULL, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to write to source array!\n");
    exit(1);
  }


  err = 0;
  err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &input);
  err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &output);
  err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &count);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to set kernel arguments! %d\n", err);
    exit(1);
  }

  err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to retrieve kernel work group info! %d\n", err);
    exit(1);
  }

  global = count;
  err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
  if (err)
  {
    printf("Error: Failed to execute kernel!\n");
    return EXIT_FAILURE;
  }

  clFinish(commands);

  err = clEnqueueReadBuffer( commands, output, CL_TRUE, 0, sizeof(float) * count, results, 0, NULL, NULL );  
  if (err != CL_SUCCESS)
  {
    printf("Error: Failed to read output array! %d\n", err);
    exit(1);
  }


  correct = 0;
  for(i = 0; i < count; i++)
  {
    if(results[i] == data[i] * data[i])
      correct++;
  }
    
  // Print a brief summary detailing the results
  //
  printf("Computed '%d/%d' correct values!\n", correct, count);
    
  // Shutdown and cleanup
  //
  clReleaseMemObject(input);
  clReleaseMemObject(output);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
 
  return 0;
  
}
