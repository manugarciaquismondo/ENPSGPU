
#ifndef APPLY_REPARTITION_PROTOCOL
#define APPLY_REPARTITION_PROTOCOL

#include "enps_parameters.h"
//#include <cutil_inline.h>
__global__ void apply_repartition_protocol(Results_type* production_function_results, Coefficient_Type* repartition_protocol_coefficients, Variable_ID* repartition_protocol_variables, Results_type* variables, Active_marker* apply_program);
#endif