#include "ENPS_Model.h"
#include "Tree_Production_Function_Display.h"


void ENPS_Model::ENPS_Instance::calculateMaxSizes(){
	max_number_of_production_function_items = number_of_programs * max_size_of_production_functions;
	max_number_of_repartition_protocol_items = number_of_programs * max_size_of_repartition_protocols;
	initializeDataStructures();
}

void ENPS_Model::ENPS_Instance::initializeDataStructures(){
	/*Initialize variables array*/
	variables = (Results_type*)malloc(number_of_variables* sizeof (Results_type));

	/*Initialize reescalate variables marker*/
	production_function_operations = (Active_marker**)malloc(max_number_of_production_function_items* sizeof (Active_marker));
	/*Initialize enzymes array*/
	enzymes = (Variable_ID*)malloc(number_of_programs* sizeof (Variable_ID));
	left_child_offset= (Offset_type**)malloc(max_number_of_production_function_items* sizeof (Offset_type));
	right_child_offset= (Offset_type**)malloc(max_number_of_production_function_items* sizeof (Offset_type));


	production_function_items = (production_function_item**)malloc(number_of_programs * sizeof(production_function_item));

	production_function_coefficients = (Coefficient_Type**)malloc(max_number_of_production_function_items*sizeof(Coefficient_Type));
	production_function_variables = (Variable_ID**)malloc(max_number_of_production_function_items*sizeof(Variable_ID));
	repartition_protocol_coefficients = (Coefficient_Type**)malloc(max_number_of_repartition_protocol_items*sizeof(Coefficient_Type));
	repartition_protocol_variables = (Variable_ID**)malloc(max_number_of_repartition_protocol_items*sizeof(Variable_ID));
	for(int i=0; i<number_of_programs; i++){
		production_function_coefficients[i] = (Coefficient_Type*)malloc(max_size_of_production_functions*sizeof(Coefficient_Type));
		production_function_variables[i] = (Variable_ID*)malloc(max_size_of_production_functions*sizeof(Variable_ID));
		repartition_protocol_coefficients[i] = (Coefficient_Type*)malloc(max_size_of_repartition_protocols*sizeof(Coefficient_Type));
		repartition_protocol_variables[i] = (Variable_ID*)malloc(max_size_of_repartition_protocols*sizeof(Variable_ID));
		production_function_operations[i] = (Active_marker*)malloc(max_size_of_production_functions*sizeof(Active_marker));
		left_child_offset[i] = (Offset_type*)malloc(max_size_of_production_functions*sizeof(Offset_type));
		right_child_offset[i] = (Offset_type*)malloc(max_size_of_production_functions*sizeof(Offset_type));

		production_function_items[i] = (production_function_item*)malloc(sizeof(production_function_item));

		/*Set the production function items to an initial null value*/
		production_function_items[i]->left_operand=0;
		production_function_items[i]->right_operand=0;
	}

	clearValues();
}

ENPS_Model::ENPS_Instance::ENPS_Instance(){}



ENPS_Model::ENPS_Instance::~ENPS_Instance(){
	free(variables);
	free(enzymes);
	free(production_function_operations);

	free_pointer_structures();

}



/*void ENPS_Test_Cases_Library::Test_Class::print_according_to_operation(int i){
	if(operations[i]==SUBSTRACT) 
		printf("%4.2f-[%i]", production_function_coefficients[i][0], production_function_variables[i][0]);
	if(operations[i]==MULTIPLY_AND_SUBSTRACT)
		printf("[%i]*(%4.2f-[%i])", production_function_variables[i][0], production_function_coefficients[i][0], production_function_variables[i][1]);
	if(operations[i]==MULTIPLY)
		printf("%4.2f*[%i]", production_function_coefficients[i][0], production_function_variables[i][0]);
}*/

void ENPS_Model::ENPS_Instance::print_enzyme(int index){
	Variable_ID enzyme= enzymes[index];
	if(enzyme!=NON_VALID_ENZYME_MASK)
		printf("\nEnzyme: %3i", enzymes[index]);
	printf("\n");
}

void ENPS_Model::ENPS_Instance::print_constants(){

	printf("P System Constants:\n");
	printf("\tNumber of programs: %i\n", number_of_programs);
	printf("\tNumber of variables: %i\n", number_of_variables);
	printf("\tMaximum size of production functions: %i\n", max_size_of_production_functions);
	printf("\tMaximum size of repartiton protocols: %i\n\n", max_size_of_repartition_protocols);
}

void ENPS_Model::ENPS_Instance::display(){

	ENPS_Model::Tree_Production_Function_Display pf_display;
	pf_display.set_test_class(this);
	print_constants();
	printf("P System Variables:\n");
	for(int i=0; i<number_of_variables; i++){
		printf("%4.2f ", variables[i]);
		printf("\n");
	}

	printf("\nP System programs:\n");
	for(int i=0; i<number_of_programs; i++){
		printf("\nProgram %i\n", i);
		printf("Production Function: \t");
		//print_according_to_operation(i);
		pf_display.arrayDisplay(i);
		printf("\nRepartition protocol: \t", i);
		for(int j=0; j<max_size_of_repartition_protocols; j++){	
			if(repartition_protocol_coefficients[i][j]==0.0f) continue;
			printf("%4.2f|[%i]+",repartition_protocol_coefficients[i][j], repartition_protocol_variables[i][j]);			
		}
		print_enzyme(i);
	}

}

void ENPS_Model::ENPS_Instance::clearValues(){
	for(int i=0; i<number_of_programs; i++){
		/*Originally, no program has associated enzyme*/
		enzymes[i]=NON_VALID_ENZYME_MASK;
		for(int j=0; j<max_size_of_production_functions; j++){
			production_function_coefficients[i][j]=0.0f;
			production_function_variables[i][j]=NON_VALID_VARIABLE;
		}
		for(int j=0; j<max_size_of_repartition_protocols; j++){
			/*Originally, no repartition protocol can be applied*/
			repartition_protocol_coefficients[i][j]=0.0f;
		}
	}
}

void ENPS_Model::ENPS_Instance::normalizeCoefficients(){
	/*The coefficient normalization is necessary in order to apply the variable contributions of the repartition protocol*/
	Coefficient_Type coefficients_summing=0.0f;
	for(int i=0; i<number_of_programs; i++){
		coefficients_summing=0.0f;
		for(int j=0; j<max_size_of_repartition_protocols; j++)
			coefficients_summing+=repartition_protocol_coefficients[i][j];
		for(int j=0; j<max_size_of_repartition_protocols; j++)
			repartition_protocol_coefficients[i][j]/=coefficients_summing;
	}
}

void ENPS_Model::ENPS_Instance::free_production_function_item(production_function_item* item){
	if(is_valid_flag(item->operation)){
		if(is_operation(item->operation)){
			free_production_function_item(item->left_operand);
			free_production_function_item(item->right_operand);
		}
		free(item);
	}

}

int ENPS_Model::ENPS_Instance::is_valid_flag(Active_marker operation){
	if(is_operation(operation)) return 1;
	switch(operation){
		case VARIABLE: return 1;
		case COEFFICIENT: return 1;
		default: return 0;
	}
	return 0;
}


int ENPS_Model::ENPS_Instance::is_operation(Active_marker operation){
	switch(operation){
		 case ADD: return 1;
		 case SUBSTRACT: return 1;
		 case POW: return 1;
		 case MULTIPLY: return 1;
		 case DIV: return 1;
		 default: return 0;
	}
}

void ENPS_Model::ENPS_Instance::free_pointer_structures(){

	/*for(int i=0; i<number_of_programs; i++){
		free(production_function_coefficients[i]);
		free(production_function_variables[i]);
		//free(production_function_operations[i]);
		//free(left_child_offset[i]);
		//free(right_child_offset[i]);
		free(repartition_protocol_coefficients[i]);
		free(repartition_protocol_variables[i]);
		free_production_function_item(production_function_items[i]);
	}*/
	free(production_function_coefficients);
	free(production_function_variables);
	//free(production_function_operations);
	/*free(left_child_offset);
	free(right_child_offset);*/
	free(repartition_protocol_coefficients);
	free(repartition_protocol_variables);
	free(production_function_items);
}