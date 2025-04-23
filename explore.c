
/*
 * dims = G*L (global_id = g*num_threads + l)
 *
 *            OpenCL        CUDA       HIP                  METAL
 * G cores   (get_group_id, blockIdx,  __ockl_get_group_id, threadgroup_position_in_grid)
 * L threads (get_local_id, threadIdx, __ockl_get_local_id, thread_position_in_threadgroup)
 *
**/

#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define SIZE 128

const char* clkernel = 
"__kernel void explore(__global float* C) { \n"
"                                       \n"
"    int i = get_local_id(0);           \n"
"    C[get_global_id(0)] = i;           \n"
"}                                      \n";

int main() {
  float *C = (float*)malloc(sizeof(float) * SIZE);
  //
  cl_int err;
  cl_platform_id platform;
  cl_device_id device;
  clGetPlatformIDs(1, &platform, NULL);
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  //
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
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
  size_t globalSize = 128;
  size_t localSize  = 4;
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
