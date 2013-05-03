#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>

#include <OpenCL/cl.h>

const char *myKernel = "\n" \
"__kernel void matrixadd( \n" \
"   __global int* matrix,\n" \
"   __global int* result,\n" \
"   int arraySize,\n" \
"   int add,\n" \
"   int mult)\n" \
"{  \n" \
"       int gid = get_global_id(0);\n" \
"       int start = arraySize * gid; \n" \
"       int end = arraySize * (gid + 1); \n" \
"       for (int i = start; i < end; i++) \n" \
"           matrix[i] = (matrix[i] + add); \n" \
"} \n" \
"\n";

const char *myKernel2 = "\n" \
"__kernel void matrixmult( \n" \
"   __global int* matrix,\n" \
"   __global int* result,\n" \
"   int arraySize,\n" \
"   int add,\n" \
"   int mult)\n" \
"{  \n" \
"       int gid = get_global_id(0);\n" \
"       int start = arraySize * gid; \n" \
"       int end = arraySize * (gid + 1); \n" \
"       for (int i = start; i < end; i++) \n" \
"           result[i] = matrix[i] * mult; \n" \
"} \n" \
"\n";

const int ARRAY_SIZE = 1000;

int main(int argc, const char * argv[])
{
    
    clock_t start, end;
    
    start = clock();
    
    cl_int errNum;
    cl_uint numPlatforms;
    cl_platform_id firstPlatformId;
    cl_context context = NULL;
    cl_device_id *devices;
    cl_device_id device;
    cl_command_queue commandQueue = NULL;
    cl_command_queue commandQueue2 = NULL;
    size_t deviceBufferSize = 1;
    cl_program program, program2;
    cl_kernel kernel = 0;
    cl_kernel kernel2 = 0;
    cl_mem matrixBuffer;
    cl_mem resultBuffer;
    
    
    // First create a context
    //      step 1 get the platform
    //      step 2 create the context properties
    //      step 3 create the context
    errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
    
    
    cl_context_properties contextProperties[] =
    {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)firstPlatformId,
        0
    };
    context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU,NULL, NULL, &errNum);
    
    
    //Second create a command queue
    //      step 1 get the device for the queue
    //      step 2 create the queue
    
    errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);
    devices = new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);
    device = devices[0];
    
    
    commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);
    commandQueue2 = clCreateCommandQueue(context, devices[0], 0, NULL);
    
    
    //Third create a program
    //      step 1 create the program with the kernel function
    //      step 2 build the program
    program = clCreateProgramWithSource(context, 1,
                                        (const char**)&myKernel ,
                                        NULL, NULL);
    program2 = clCreateProgramWithSource(context, 1,
                                         (const char**)&myKernel2,
                                         NULL, NULL);
    
    errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    
    errNum = clBuildProgram(program2, 0, NULL, NULL, NULL, NULL);
    if (errNum != CL_SUCCESS)
    {
        // Determine the reason for the error
        char buildLog[16384];
        clGetProgramBuildInfo(program2, device, CL_PROGRAM_BUILD_LOG,
                              sizeof(buildLog), buildLog, NULL);
        
        std::cerr << "Error in kernel: " << std::endl;
        std::cerr << buildLog;
        clReleaseProgram(program);
        return NULL;
    }
    
    //Fourth create the Kernel object
    kernel = clCreateKernel(program, "matrixadd", NULL);
    kernel2 = clCreateKernel(program2, "matrixmult", NULL);
    
    
    //Fifth Set up memory
    //      step 1 - set up host memory variables
    //      step 2 - create OpenCL buffers
    int matrix[ARRAY_SIZE][ARRAY_SIZE];
    int result[ARRAY_SIZE * ARRAY_SIZE];
    
    srand(time(NULL));
    for (int i = 0; i < ARRAY_SIZE; i++)
        for (int j = 0; j < ARRAY_SIZE; j++)
            matrix[i][j] = rand() % 10;
    
    matrixBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,sizeof(int) * ARRAY_SIZE * ARRAY_SIZE, matrix, NULL);
    
    resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,sizeof(int) * ARRAY_SIZE * ARRAY_SIZE, NULL , NULL);
    
    int add = 5;
    std::cout << "Input Number to add: " << std::endl;
    std::cin >> add;
    
    int mult = 5;
    std::cout << "Input Number to multiply: " << std::endl;
    std::cin >> mult;
    
    std::cout << " " << std::endl;
    
    //Sixth Set up the Kernel arguments
    errNum = clSetKernelArg(kernel, 0, sizeof(cl_mem), &matrixBuffer);
    errNum = clSetKernelArg(kernel, 1, sizeof(cl_mem), &resultBuffer);
    errNum = clSetKernelArg(kernel, 2, sizeof(int), &ARRAY_SIZE);
    errNum = clSetKernelArg(kernel, 3, sizeof(int), &add);
    errNum = clSetKernelArg(kernel, 4, sizeof(int), &mult);
    
    errNum = clSetKernelArg(kernel2, 0, sizeof(cl_mem), &matrixBuffer);
    errNum = clSetKernelArg(kernel2, 1, sizeof(cl_mem), &resultBuffer);
    errNum = clSetKernelArg(kernel2, 2, sizeof(int), &ARRAY_SIZE);
    errNum = clSetKernelArg(kernel2, 3, sizeof(int), &add);
    errNum = clSetKernelArg(kernel2, 4, sizeof(int), &mult);
    
    //Seventh Queue the kernel for execution on the device
    //      step 1 - set up the work items and group
    size_t workItems[1] = { ARRAY_SIZE };
    size_t groupSize[1] = { 1 };
    
    cl_event event;
    cl_event event2;
    
    errNum = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL,workItems, groupSize, 0, NULL, &event);
    
    if (errNum != CL_SUCCESS)
        std::cout << "Could not queue the first kernel\n";
    
    errNum = clEnqueueNDRangeKernel(commandQueue2, kernel2, 1, NULL, workItems, groupSize, 1, &event, &event2);
    
    if (errNum != CL_SUCCESS)
        std::cout << "Could not queue the second kernel\n";
    
    //Eighth Read the results in any results buffers that were changed
    errNum = clEnqueueReadBuffer(commandQueue2, resultBuffer, CL_TRUE,0, ARRAY_SIZE * ARRAY_SIZE * sizeof(int), result, 1, &event2, NULL);
    
    if (errNum != CL_SUCCESS)
        std::cout << "Could not read results";
    
    //output to file
    std::ofstream myfile;
    myfile.open("result.txt");
    
    //output the matrix
    myfile << "============= ORIGINAL MATRIX ===============" << std::endl;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        for (int j = 0; j < ARRAY_SIZE; j++)
            myfile << matrix[i][j] << " ";
        myfile << "\n";
    }
    
    myfile << " " << std::endl;
    
    // Output the result buffer
    myfile << "============= RESULT MATRIX ===============" << std::endl;
    for (int i = 0; i < ARRAY_SIZE * ARRAY_SIZE; i++)
    {
        if(i % ARRAY_SIZE == 0 && i != 0) myfile << std::endl;
        myfile << result[i] << " ";
    }
    
    myfile.close();
    
    std::cout << "Executed program succesfully." << std::endl;
    
    end = clock() - start;
    
    std::cout << "TIME: ";
    std::cout << (double)end / (double)CLOCKS_PER_SEC;
    std::cout << " seconds" << std::endl;
    
}

