#ifndef COMPUTE_PRODUCTION_FUNCTION
#define COMPUTE_PRODUCTION_FUNCTION
#include "enps_parameters.h"
#include <math.h>
#include "stack_constants.cuh"
__global__ void compute_production_function(Offset_type *left_child_offset, Offset_type *right_child_offset, Coefficient_Type* production_function_coefficients, Variable_ID* production_function_variables, Results_type* variables, Results_type* production_function_results, Active_marker* apply_programs, Active_marker* production_function_operations, int production_function_size, int number_of_programs);

#endif