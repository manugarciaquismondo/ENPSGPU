#include "apply_repartition_protocol.cuh"

//__device__ void atomicAddf(float* address, float value){
   // #if __CUDA_ARCH__ >= 200 // for Fermi, atomicAdd supports floats
       
    /*#elif __CUDA_ARCH__ >= 110
        // float-atomic-add from
        float old = value;
        while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
    #endif*/
//} 

__global__ void apply_repartition_protocol(Results_type* production_function_results, Coefficient_Type* repartition_protocol_coefficients, Variable_ID* repartition_protocol_variables, Results_type* variables, Active_marker* apply_program){
	__syncthreads();
	int programIdx = blockIdx.x;
	Results_type pf_result = production_function_results[programIdx];
	/*Check if the program is to be applied*/
	if(pf_result==0||!apply_program[programIdx]) return;

	/*Check if the repartition protocol contribution is to be applied*/
	int itemIdx = programIdx*blockDim.x + threadIdx.x;
	Coefficient_Type coefficient = repartition_protocol_coefficients[itemIdx];
	if(coefficient==0) return;

	Variable_ID variableIdx = repartition_protocol_variables[itemIdx];
	Results_type contribution = pf_result*coefficient;
	 //atomicAdd(address,value);
	atomicAdd(&(variables[variableIdx]), contribution);
	__syncthreads();

}