
#include "Tree_Production_Function_Display.h"
#include <iostream>
#include <sstream>
#include <string>

using namespace std;


ENPS_Model::Tree_Production_Function_Display::Tree_Production_Function_Display(void)
{
}

ENPS_Model::Tree_Production_Function_Display::~Tree_Production_Function_Display(void)
{
}


void ENPS_Model::Tree_Production_Function_Display::set_test_class(ENPS_Model::ENPS_Instance *ENPS){
	this->ENPS = ENPS;
}
void ENPS_Model::Tree_Production_Function_Display::display_production_function(int i){
	cout << display_production_function_item(ENPS->production_function_items[i]);
}

string ENPS_Model::Tree_Production_Function_Display::display_production_function_item(production_function_item *pf_item){
	Active_marker operation = pf_item->operation;
	ostringstream auxiliary_output_stream;
	if(operation==COEFFICIENT){
		auxiliary_output_stream << pf_item->coefficient;
		return auxiliary_output_stream.str();
	}
	if(operation==VARIABLE){
		auxiliary_output_stream << "[" << pf_item->variable << "]";
		return auxiliary_output_stream.str();
	}
	string message ="(";
	message+=display_production_function_item(pf_item->left_operand);
	if(operation==ADD)
		message+="+";
	if(operation==MULTIPLY)
		message+="*";
	if(operation==POW)
		message+="^";
	if(operation==SUBSTRACT)
		message+="-";
	if(operation==DIV)
		message+="/";
	message+=display_production_function_item(pf_item->right_operand);
	message+=")";
	return message;
		
}

std::string ENPS_Model::Tree_Production_Function_Display::array_display_production_function(int program, Offset_type element){
	Active_marker operation = ENPS->production_function_operations[program][element];
	ostringstream auxiliary_output_stream;
	if(operation==COEFFICIENT){
		auxiliary_output_stream << ENPS->production_function_coefficients[program][element];
		return auxiliary_output_stream.str();
	}
	if(operation==VARIABLE){
		auxiliary_output_stream << "[" << ENPS->production_function_variables[program][element] << "]";
		return auxiliary_output_stream.str();
	}
	string message ="(";
	message+=array_display_production_function(program, ENPS->left_child_offset[program][element]);
	if(operation==ADD)
		message+="+";
	if(operation==MULTIPLY)
		message+="*";
	if(operation==POW)
		message+="^";
	if(operation==SUBSTRACT)
		message+="-";
	if(operation==DIV)
		message+="/";
	message+=array_display_production_function(program, ENPS->right_child_offset[program][element]);
	message+=")";
	return message;

}

void ENPS_Model::Tree_Production_Function_Display::arrayDisplay(int program){
	cout << array_display_production_function(program, 0);

}