#ifndef PRODUCTION_FUNCTION_ITEM
#define PRODUCTION_FUNCTION_ITEM
#include "enps_parameters.h"


typedef struct production_function_item{
	Active_marker operation;
	struct production_function_item* left_operand;
	struct production_function_item* right_operand;
	Variable_ID variable;
	Coefficient_Type coefficient;

} production_function_item;
#endif

