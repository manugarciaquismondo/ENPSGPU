
#include "production_function_solver.h"
#include <math.h>

ENPS_Production_Function_Solver::Production_Function_Solver::Production_Function_Solver(void)
{
}

ENPS_Production_Function_Solver::Production_Function_Solver::~Production_Function_Solver(void)
{
}

Results_type ENPS_Production_Function_Solver::Production_Function_Solver::solve(production_function_item *item, Results_type* variables){
	Active_marker item_type = item->operation;
	if(item_type==COEFFICIENT)
		return item->coefficient;
	if(item_type==VARIABLE)
		return variables[item->variable];
	if(item_type==ADD)
		return solve(item->left_operand, variables) + solve(item->right_operand, variables);
	if(item_type==SUBSTRACT)
		return solve(item->left_operand, variables) - solve(item->right_operand, variables);
	if(item_type==MULTIPLY)
		return solve(item->left_operand, variables) * solve(item->right_operand, variables);
	if(item_type==DIV)
		return solve(item->left_operand, variables) * solve(item->right_operand, variables);
	if(item_type==POW)
		return pow((float)solve(item->left_operand, variables), (float)solve(item->right_operand, variables));
	return 0.0f;
}