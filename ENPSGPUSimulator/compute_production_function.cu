#include "compute_production_function.cuh"

__device__ Results_type compute_production_function_element(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Active_marker* production_function_operations, int program_base, int itemIdx);
__device__ unsigned char is_valid_operation(Active_marker operation);
__device__ void compute_production_function_non_compatible(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs);
__device__ void compute_production_function_backwards_compatible(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs);

__device__ Results_type get_constant_variable_or_known_operation(
			Variable_ID position, 
			Active_marker* production_function_operations, 
			Variable_ID* production_function_variables, 
			Coefficient_Type* production_function_coefficients, 
			int temporalIndex, 
			Results_type* left_values, 
			Results_type* right_values);

__device__ void calculate_result(	
	Results_type* left_values,
	Results_type* right_values,
	Active_marker* storage,
	Variable_ID* position,
	Variable_ID* storagePosition,
	Offset_type *left_child_offset, 
	Offset_type *right_child_offset, 
	Coefficient_Type* production_function_coefficients, 
	Variable_ID* production_function_variables, 
	Results_type* variables,
	Active_marker* production_function_operations,
	int* index,
	Results_type* production_function_results);

__device__ char operators_calculated(Active_marker marker, Active_marker storage_marker);

__global__ void compute_production_function(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs){
	//#if __CUDA_ARCH__ >=200
		compute_production_function_non_compatible(left_child_offset, right_child_offset, production_function_coefficients, production_function_variables, variables, production_function_results, apply_programs, production_function_operations, production_function_size, number_of_programs);
	/*#else
		compute_production_function_backwards_compatible(left_child_offset, right_child_offset, production_function_coefficients, production_function_variables, variables, production_function_results, apply_programs, production_function_operations, production_function_size, number_of_programs);

	#endif*/
}


__device__ void compute_production_function_non_compatible(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs){
	
	__syncthreads();
	
	/* Access a production function item of a production function in a membrane. Thus, both dimensions of the block need to be used. That is, blockIdx.y goes for the membrane, blockIdx.x goes for the production function and threadIdx.x goes for the production function item*/
	int programIdx = blockIdx.x*blockDim.x+threadIdx.x;
	if(programIdx>=number_of_programs) return;
	/*Check if the program is to be applied*/
	if(!apply_programs[programIdx]) return;
	int program_base = production_function_size*programIdx;
	/*Set the element calculation result to the program result*/

	production_function_results[programIdx] = compute_production_function_element(left_child_offset, right_child_offset, production_function_coefficients, production_function_variables, variables, production_function_operations, program_base, 0);

	__syncthreads();

}

__device__ Results_type compute_production_function_element(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Active_marker* production_function_operations, int program_base, int itemIdx){
	int local_index = program_base + itemIdx;
	Active_marker operation = production_function_operations[local_index];
	if(operation==VARIABLE)
		return variables[production_function_variables[local_index]];
	if(operation==COEFFICIENT)
		return production_function_coefficients[local_index];
	Variable_ID lc_index = left_child_offset[local_index];
	Variable_ID rc_index = right_child_offset[local_index];
	Results_type left_result;
	Results_type right_result;
	left_result = compute_production_function_element(left_child_offset, right_child_offset, production_function_coefficients, production_function_variables, variables, production_function_operations, program_base, lc_index);
	right_result = compute_production_function_element(left_child_offset, right_child_offset, production_function_coefficients, production_function_variables, variables, production_function_operations, program_base, rc_index);
	if(operation==ADD)
		return left_result+right_result;
	if(operation==MULTIPLY)
		return left_result*right_result;
	if(operation==SUBSTRACT)
		return left_result- right_result;
	if(operation==DIV)
		return left_result/right_result;
	if(operation==POW)
		return (Results_type)pow((Coefficient_Type)left_result, (int)right_result);
	return 0;


}

__device__ unsigned char is_valid_operation(Active_marker operation){
	if(operation==VARIABLE||operation==COEFFICIENT) return 1;
	if(operation==ADD||
		operation==SUBSTRACT||
		operation==MULTIPLY||
		operation==DIV||
		operation==POW)
		return 1;
	return 0;
}




__device__ void compute_production_function_backwards_compatible(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs){
	__shared__ Results_type* left_values;
	__shared__ Results_type* right_values;
	__shared__ Active_marker* storage;
	__shared__ Variable_ID* storagePosition;
	__shared__ Variable_ID* position;
	__syncthreads();

	/* Access a production function item of a production function in a membrane. Thus, both dimensions of the block need to be used. That is, blockIdx.y goes for the membrane, blockIdx.x goes for the production function and threadIdx.x goes for the production function item*/
	int programIdx = blockIdx.x * blockDim.x + threadIdx.x;
	if(programIdx>=number_of_programs) return;

	/*Check if the program is to be applied*/
	if(!apply_programs[programIdx]) return;
	int program_base = production_function_size*programIdx;
	storage[0]=STACK_RESULT_VALUE;
	position[0] = program_base;
	int index=0;

	while((index)!=-1){
		calculate_result(
			left_values,
			right_values,
			storage,
			position,
			storagePosition,
			left_child_offset,
			right_child_offset,
			production_function_coefficients,
			production_function_variables,
			variables,
			production_function_operations,
			&index,
			production_function_results
			);
	
	}
	/*Set the element calculation result to the program result*/

	
	__syncthreads();

}

__device__ void calculate_result(	
	Results_type* left_values,
	Results_type* right_values,
	Active_marker* storage,
	Variable_ID* position,
	Variable_ID* storagePosition,
	Offset_type *left_child_offset, 
	Offset_type *right_child_offset, 
	Coefficient_Type* production_function_coefficients, 
	Variable_ID* production_function_variables, 
	Results_type* variables,
	Active_marker* production_function_operations,
	int* index,
	Results_type* production_function_results){

	Results_type result;
	int temporalIndex= *index;
	if(operators_calculated(production_function_operations[position[temporalIndex]], storage[temporalIndex])){
		result=get_constant_variable_or_known_operation(position[temporalIndex], 
			production_function_operations, 
			production_function_variables, 
			production_function_coefficients, 
			temporalIndex, 
			left_values, 
			right_values);
		switch(storage[temporalIndex]){
		case(STACK_LEFT_VALUE):
			left_values[storagePosition[temporalIndex]]=result;
			break;
		case(STACK_RIGHT_VALUE):
			right_values[storagePosition[temporalIndex]]=result;
			break;
		default:
			production_function_results[position[temporalIndex]]=result;
		}
		(*index)--;

	}
	else{

		(*index)++;
		storage[*index]=STACK_LEFT_VALUE;
		storage[temporalIndex]|=LEFT_OPERAND_CALCULATED;
		storagePosition[*index]=temporalIndex;
		position[*index]=left_child_offset[position[temporalIndex]]+ position[temporalIndex];

		(*index)++;
		storage[*index]=STACK_RIGHT_VALUE;
		storage[temporalIndex]|=RIGHT_OPERAND_CALCULATED;
		storagePosition[*index]=temporalIndex;
		position[*index]=right_child_offset[position[temporalIndex]]+ position[temporalIndex];
	}
}



__device__ float get_constant_variable_or_known_operation(
	Variable_ID position, 
	Active_marker* production_function_operations, 
	Variable_ID* production_function_variables, 
	Coefficient_Type* production_function_coefficients,
	int index,
	Results_type* left_values,
	Results_type* right_values
	){
	Active_marker operation = production_function_operations[position];
	if(operation==COEFFICIENT)
		return production_function_coefficients[position];
	if(operation==VARIABLE)
		return production_function_variables[position];
	Results_type left_value = left_values[index];
	Results_type right_value = right_values[index];
	if(operation==ADD)
		return left_value+right_value;
	if(operation==SUBSTRACT)
		return left_value-right_value;
	if(operation==MULTIPLY)
		return left_value*right_value;
	if(operation==DIV)
		return left_value/right_value;
	if(operation==POW)
		return pow(left_value,right_value);
	return 0.0f;
		/*Ver si los valores izquierdos y derechos estan resueltos. Si lo estan, calcular la operacion. Si no, informar de que aun quedan por calcular*/
}


__device__ char operators_calculated(Active_marker marker, Active_marker storage_marker){
char result = marker==COEFFICIENT||marker==VARIABLE;
return result||(storage_marker&RIGHT_OPERAND_CALCULATED)&&
	(storage_marker&LEFT_OPERAND_CALCULATED);
}