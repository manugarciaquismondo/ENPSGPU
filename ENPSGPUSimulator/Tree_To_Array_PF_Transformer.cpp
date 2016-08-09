
#include "Tree_To_Array_PF_Transformer.h"
#include "ENPS_Model.h"


ENPS_XML_Reader::Tree_To_Array_PF_Transformer::Tree_To_Array_PF_Transformer(void)
{
}

ENPS_XML_Reader::Tree_To_Array_PF_Transformer::~Tree_To_Array_PF_Transformer(void)
{
}




void ENPS_XML_Reader::Tree_To_Array_PF_Transformer::transform_to_production_function_items_array(ENPS_Model::ENPS_Instance *example)
{
	int number_of_programs = example->number_of_programs;
	int pf_items_max = example->max_size_of_production_function_tree;
	for(int i=0; i<number_of_programs; i++){
		auxiliary_index=-1;
		allocate_item(*(example->production_function_items[i]), example, i);
	}
		
}

Offset_type ENPS_XML_Reader::Tree_To_Array_PF_Transformer::allocate_item(production_function_item pf_item, ENPS_Model::ENPS_Instance *ENPS, int program){
	Offset_type temporary_auxiliary_index = auxiliary_index;
	//int item_index = program*pf_items_max+auxiliary_index;
	auxiliary_index++;
	Offset_type incremented_auxiliary_index = auxiliary_index;
	copy_production_function_item(ENPS, pf_item, program);
	//resulting_items[item_index]=*pf_item;

	if(pf_item.operation!=COEFFICIENT&&pf_item.operation!=VARIABLE){
		ENPS->left_child_offset[program][incremented_auxiliary_index] = allocate_item(*pf_item.left_operand, ENPS, program);
		ENPS->right_child_offset[program][incremented_auxiliary_index] = allocate_item(*pf_item.right_operand, ENPS, program);
	}
	return incremented_auxiliary_index;


}
void ENPS_XML_Reader::Tree_To_Array_PF_Transformer::copy_production_function_item(ENPS_Model::ENPS_Instance *ENPS, production_function_item source_copy, int program)
{
	ENPS->production_function_coefficients[program][auxiliary_index] = source_copy.coefficient;
	ENPS->production_function_variables[program][auxiliary_index] = source_copy.variable;
	ENPS->production_function_operations[program][auxiliary_index]=source_copy.operation;
}