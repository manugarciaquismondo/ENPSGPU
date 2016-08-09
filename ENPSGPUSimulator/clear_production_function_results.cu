#include "clear_production_function_results.cuh"

__global__ void clear_production_function_results(Results_type* production_function_results, Active_marker* apply_programs, Variable_ID* enzymes, Results_type* variables, Enzyme_type* required_enzymes, int number_of_programs){
	__syncthreads();
	int programIdx = threadIdx.x;
	if(programIdx>=number_of_programs) return;
	production_function_results[programIdx]=0;
	/*Set if the program is to be applied*/
	Variable_ID enzyme_ID = enzymes[programIdx];
	/*If the program has an enzyme whose value is lower than the required value, do not apply the program. Otherwise, apply it*/
	if((enzyme_ID!=NON_VALID_ENZYME_MASK)&&((Enzyme_type)variables[enzyme_ID])<=required_enzymes[programIdx])
		apply_programs[programIdx]=NOT_APPLY_PROGRAM;
	else
		apply_programs[programIdx]=APPLY_PROGRAM;
	__syncthreads();
}