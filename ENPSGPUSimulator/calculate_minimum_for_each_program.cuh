#ifndef CALCULATE_MINIMUM_FOR_EACH_PROGRAM
#define CALCULATE_MINIMUM_FOR_EACH_PROGRAM
#include <math.h>
#include "enps_parameters.h"
__global__ void calculate_minimum_for_each_program(Active_marker *production_function_operations, Variable_ID* enzymes, Variable_ID* production_function_variables, Enzyme_type* required_enzymes, Results_type* variables, int production_function_size, int number_of_programs);
#endif