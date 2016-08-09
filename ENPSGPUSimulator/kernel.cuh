#ifndef CUDA_KERNEL_ENPS
#define CUDA_KERNEL_ENPS
#include "cuda_runtime.h"
#include "enps_parameters.h"
#include "device_launch_parameters.h"
#include "ENPS_Model.h"
#include "ENPS_XML_Reader.h"
#include "Results_Printer.h"
#include "Tree_To_Array_PF_Transformer.h"

#include "kernel_function_headers.cuh"


#include <stdio.h>

#define BLOCK_SIZE 256

using namespace ENPS_Model;
using namespace std;


cudaError_t computeENPSsimulation(ENPS_Instance* example, char* resultsRoute, int iterations);
void flattenElements(ENPS_Instance* example);
int performHostDataMemallocs(ENPS_Instance *example);
int performHostDataMemcpys(ENPS_Instance *example);
int performDeviceDataMemallocs(ENPS_Instance *example);
int copySimulationResultsToHostMemory(int number_of_variables);

int perform_cuda_malloc_and_check_error(cudaError_t status);
extern "C" int simulateENPS(char* simulatorRoute, char* resultsRoute, int iterations, char show_results);
cudaError_t computeENPSsimulation(ENPS_Instance* example, int iterations);
void write_results(int number_of_variables, char* resultsRoute, int cycles, int max_number_of_rules, float elapsedTime);
void write_variables(FILE* resultsFile, int number_of_variables);
void allocate_variable_names(int number_of_variables);
void freeGPUPointers();
void freeHostPointers();

#endif
