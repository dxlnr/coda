#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 128

const char* clkernel = 
"__kernel void explore(__global float* C) { \n"
"                                           \n"
"    int i = get_local_id(0);               \n"
"    C[get_global_id(0)] = i;               \n"
"}                                          \n";

int main() {
  float *C = (float*)malloc(sizeof(float) * SIZE);
  // INITIATE COMPUTE SYSTEM
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  // 
  cl_queue_properties props[] = {
    CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
    0
  };
  cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, props, &err);
  if (err != CL_SUCCESS) { printf("Failed to create command queue. Error code: %d\n", err); return 1; }
  // BUILD PROGRAM & KERNEL
  cl_program program = clCreateProgramWithSource(context, 1, &clkernel, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "explore", NULL);
  // 
  cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, SIZE * sizeof(float), NULL, NULL);
  //
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufC);
  // KERNEL EXECUTION
  size_t globalSize = 128; // total number of threads (G*L)
  size_t localSize  = 32;  // threads per work-group (warp) (32 is max)
  //
  cl_event kernel_event;
  /*
   * clEnqueueNDRangeKernel
   * ----------------------
   * cl_command_queue command_queue,
   * cl_kernel        kernel,
   * cl_uint          work_dim,
   * const size_t *   global_work_offset,
   * const size_t *   global_work_size,
   * const size_t *   local_work_size,
   * cl_uint          num_events_in_wait_list,
   * const cl_event * event_wait_list,
   * cl_event *       event) CL_API_SUFFIX__VERSION_1_0;
  */
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, NULL);
  // Ensure the kernel finishes before measuring
  clWaitForEvents(1, &kernel_event);

  // MEASURE KERNEL RUNTIME
  cl_ulong time_start, time_end;
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(time_start), &time_start, NULL);
  clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(time_end), &time_end, NULL);

  double kernel_runtime_ms = (time_end - time_start) * 1e-6; // ns → ms
  printf("Runtime: %.3f ms\n", kernel_runtime_ms);
  // READ RESULT
  clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, SIZE * sizeof(float), C, 0, NULL, NULL);
  // PRINT RESULT
  for (int i = 0; i < SIZE; i++) {
    if (i % 16 == 0 && i != 0) {
      printf("\n");
    }
    printf(" %3d", (int)C[i]);
  } 
  printf("\n");
  // CLEANUP
  clReleaseMemObject(bufC);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  free(C); 

  return 0;
}
