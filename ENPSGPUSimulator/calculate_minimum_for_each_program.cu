#include "calculate_minimum_for_each_program.cuh"

__global__ void calculate_minimum_for_each_program(Active_marker* production_function_operations, Variable_ID* enzymes, Variable_ID* production_function_variables, Enzyme_type* required_enzymes, Results_type* variables, int production_function_size, int number_of_programs){
	__syncthreads();
	int programIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(programIdx>=number_of_programs) return;
	/*If there is no enzyme associated to the program, return*/
	Variable_ID enzyme_ID = enzymes[programIdx];
	if(enzyme_ID==NON_VALID_ENZYME_MASK) return;
	Results_type local_minimum=ENZYME_MAX_VALUE;
	int pf_variableIdx= production_function_size*programIdx;
	int pf_upper_bound = pf_variableIdx+production_function_size;
	for(int i=pf_variableIdx; i<pf_upper_bound; i++)
		if(production_function_operations[i]==VARIABLE){
			local_minimum=min(local_minimum, variables[production_function_variables[i]]);
		}
	required_enzymes[programIdx] = local_minimum;
	__syncthreads();


}