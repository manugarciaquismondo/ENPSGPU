#ifndef PRODUCTION_FUNCTION_SOLVER
#define PRODUCTION_FUNCTION_SOLVER
#include "production_function_item.h"
namespace ENPS_Production_Function_Solver{
	class Production_Function_Solver
	{
	public:
		Production_Function_Solver(void);
		~Production_Function_Solver(void);
		Results_type solve(production_function_item *item, Results_type* variables);
	};
};
#endif