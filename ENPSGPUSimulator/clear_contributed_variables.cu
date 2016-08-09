#include "clear_contributed_variables.cuh"



__global__ void clear_contributed_variables(Active_marker* production_function_operations, Variable_ID* production_function_variables, Results_type* variables, Active_marker* apply_programs){
	__syncthreads();
	int programIdx = blockIdx.x;
	/*Check if the program is to be applied*/
	if(!apply_programs[programIdx]) return;

	/*Check if the specific operation is a variable*/
	int itemIdx = programIdx*blockDim.x+ threadIdx.x;
	if(production_function_operations[itemIdx]!=VARIABLE) return;
	Variable_ID variableIdx = production_function_variables[itemIdx];
	variables[variableIdx]=0.0f;
	__syncthreads();

}