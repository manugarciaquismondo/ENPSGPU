#ifndef CLEAR_PRODUCTION_FUNCTION_RESULTS
#define CLEAR_PRODUCTION_FUNCTION_RESULTS
#include "enps_parameters.h"
__global__ void clear_production_function_results(Results_type* production_function_results, Active_marker* apply_programs, Variable_ID* enzymes, Results_type* variables, Enzyme_type* required_enzymes, int number_of_programs);

#endif