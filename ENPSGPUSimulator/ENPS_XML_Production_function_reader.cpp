
#include "ENPS_XML_Production_function_reader.h"

ENPS_XML_Reader::ENPS_XML_Production_function_reader::ENPS_XML_Production_function_reader(void)
{
}

ENPS_XML_Reader::ENPS_XML_Production_function_reader::~ENPS_XML_Production_function_reader(void)
{
}

production_function_item* ENPS_XML_Reader::ENPS_XML_Production_function_reader::readProductionFunctionElement(xml_node<>* treeElement){
	std::string tree_element_name = treeElement->name();
	temp_production_function_tree_size=0;
	return readTreeElement(treeElement->first_node("math")->first_node());	
}

production_function_item* ENPS_XML_Reader::ENPS_XML_Production_function_reader::readTreeElement(xml_node<>* treeElement){
	production_function_item *resultItem=(production_function_item*)malloc(sizeof(production_function_item));
	std::string element_name = treeElement->name();
	temp_production_function_tree_size++;
	if(element_name=="cn"){
		std::string coefficient_string_name = treeElement->first_node()->value();
		Coefficient_Type coefficient = atof(coefficient_string_name.c_str());
		resultItem->coefficient = coefficient;
		resultItem->operation = COEFFICIENT;
		return resultItem;
	}
	if(element_name=="ci"){
		std::string variable_string_name = treeElement->first_node()->value();
		resultItem->variable =variableAssociations[variable_string_name];
		resultItem->operation = VARIABLE;
		return resultItem;
	}
	xml_node<>* readElement = treeElement->first_node();
	element_name = readElement->name();
	if(element_name=="add")
		resultItem->operation = ADD;
	if(element_name=="pow")
		resultItem->operation= POW;
	if(element_name=="minus")
		resultItem->operation = SUBSTRACT;
	if(element_name=="times")
		resultItem->operation = MULTIPLY;
	if(element_name=="div")
		resultItem->operation = DIV;
	xml_node<>* firstXMLOperand = readElement->next_sibling();
	xml_node<>* secondXMLOperand = firstXMLOperand->next_sibling();
	resultItem->left_operand = readTreeElement(firstXMLOperand);
	resultItem->right_operand = readTreeElement(secondXMLOperand);
	return resultItem;
}


/*void ENPS_XML_Reader::ENPS_XML_Production_function_reader::fillInProductionFunction(xml_node<> *productionVariable){
	production_function_counter=0;
	xml_node<>* mathMLnode = productionVariable->first_node("math");
	mathMLnode = mathMLnode->first_node("apply");
	readProductionFunctionVariable(mathMLnode);
	xml_node<>* reserve_pf_variable = common_production_function_variable;
	std::string elementName = reserve_pf_variable->name();
	if (elementName=="times"){	
		/*If the main operation of the production function is a multiplication*/
		//xml_node<>* secondMultiplicationOperator = common_production_function_variable->next_sibling("apply");
		//if(secondMultiplicationOperator==0){
			/*If the production function is only a multiplication*/
			/*secondMultiplicationOperator = common_production_function_variable->next_sibling("cn");
			readProductionFunctionCoefficient(secondMultiplicationOperator);
			production_function_counter++;
			ENPS->operations[program_counter] = MULTIPLY;			
		}
		else{*/
			/*If there is a substraction within the second element of the multiplication*/
			/*xml_node<>* minusNode = secondMultiplicationOperator->first_node("minus");
			production_function_counter++;
			readProductionFunctionVariable(secondMultiplicationOperator);
			xml_node<>* substractionCoefficient = minusNode->next_sibling("cn");
			production_function_counter--;
			readProductionFunctionCoefficient(substractionCoefficient);
			ENPS->operations[program_counter]= MULTIPLY_AND_SUBSTRACT;
		}
	}	*/
	/*If the main operation of the production function is a substraction*/
	/*else if (elementName=="minus"){
		readProductionFunctionCoefficient(reserve_pf_variable->next_sibling("cn"));
		production_function_counter++;
		ENPS->operations[program_counter] = SUBSTRACT;
	}
}*/

void ENPS_XML_Reader::ENPS_XML_Production_function_reader::readProductionFunctionVariable(xml_node<>* first_pf_element){
	xml_node<>* firstMultiplicationOperator = first_pf_element->first_node("ci");
	xml_node<>* commonVariableNode = firstMultiplicationOperator->first_node();
	std::string commonVariableStringValue = commonVariableNode->value();
	ENPS->production_function_variables[program_counter][production_function_counter]=variableAssociations[commonVariableStringValue];	

	/*locate the production function attribute, to be properly read by future operations*/
	common_production_function_variable =first_pf_element->next_sibling();
	if(common_production_function_variable==0) 
		common_production_function_variable=first_pf_element->first_node();
}


void ENPS_XML_Reader::ENPS_XML_Production_function_reader::readProductionFunctionCoefficient(xml_node<>* coefficientElement){
	xml_node<>* coefficientName = coefficientElement->first_node();
	std::string coefficientStringValue = coefficientName->value();
	ENPS->production_function_coefficients[program_counter][production_function_counter] =(float)atof((char*)coefficientStringValue.c_str());
}





void ENPS_XML_Reader::ENPS_XML_Production_function_reader::setVariableAssociations(std::map <std::string, Variable_ID> variableAssociations){
	this->variableAssociations = variableAssociations;
}

void ENPS_XML_Reader::ENPS_XML_Production_function_reader::setENPS(ENPS_Model::ENPS_Instance* ENPS){
	this->ENPS = ENPS;
}

void ENPS_XML_Reader::ENPS_XML_Production_function_reader::setProgramCounter(int program_counter){
	this->program_counter = program_counter;
}