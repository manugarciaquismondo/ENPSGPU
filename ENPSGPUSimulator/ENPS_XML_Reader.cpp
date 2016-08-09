#include "ENPS_XML_Reader.h"


using namespace rapidxml;
using namespace std;
using namespace ENPS_Model;


ENPS_XML_Reader::ENPS_XML_Reader::ENPS_XML_Reader(){}
ENPS_XML_Reader::ENPS_XML_Reader::~ENPS_XML_Reader(){}

long ENPS_XML_Reader::ENPS_XML_Reader::getCycles(){
	return cycles;
}

void ENPS_XML_Reader::ENPS_XML_Reader::readCycles(xml_node<>* cyclesNode){
	cycles=0;
	if (cyclesNode!=NULL)
		cycles= atol(cyclesNode->value());
}

void ENPS_XML_Reader::ENPS_XML_Reader::readENPSDescription(xml_document<>* source){
	xml_node<>* membraneSystem = source->first_node("membraneSystem");
	readCycles(membraneSystem->first_node("cycles"));
	xml_node<>* parameters = membraneSystem->first_node("parameters");
	variable_counter=0;
	program_counter=0;
	fillInParameters(parameters);
	xml_node<>* currentMembrane = membraneSystem->first_node("membrane");

	read_mode=READ_VARIABLES;
	/*The membrane tree is read recursively, starting from the root*/
	fillInMembrane(currentMembrane);
	read_mode=READ_PROGRAMS;
	fillInMembrane(currentMembrane);



}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInParameters(xml_node<>* parameters){
	ENPS =(ENPS_Instance*)malloc(sizeof(ENPS_Instance));
	/*Read ENPS parameters from the XML node*/
	xml_attribute<>* programs = parameters->first_attribute("programs");
	xml_attribute<>* variables = parameters->first_attribute("variables");
	xml_attribute<>* maxRPSize = parameters->first_attribute("maxRPSize");
	xml_attribute<>* maxPFSize = parameters->first_attribute("maxPFSize");

	/*Cast programs values from programs attributes*/
	int programsValue =  atoi(programs->value());
	int variablesValue = atoi(variables->value());
	int maxRPSizeValue = atoi(maxRPSize->value());
	int maxPFSizeValue = atoi(maxPFSize->value());

	/*Set ENPS parameters*/
	ENPS->number_of_programs = programsValue;
	ENPS->number_of_variables = variablesValue;
	ENPS->max_size_of_repartition_protocols = maxRPSizeValue;
	ENPS->max_size_of_production_functions = maxPFSizeValue;
	ENPS->calculateMaxSizes();

}


ENPS_Model::ENPS_Instance* ENPS_XML_Reader::ENPS_XML_Reader::readENPSFile(char* route){
	ENPS= new ENPS_Model::ENPS_Instance;
	ifstream file;
	string file_buffer, file_content;
	file.open(route);
	ENPS->max_size_of_production_function_tree=0;
	xml_document<> document;
	/*Fill in the buffer to be parsed as an XML document*/
	if(file.is_open()){
		while(!file.eof()){
			file >> file_buffer;
			if(!file.eof()){
				file_content.append(file_buffer);
				file_content.append("\n");
			}
			
		}
		document.parse<0>((char*)file_content.c_str());
		readENPSDescription(&document);
		ENPS->normalizeCoefficients();
	}

	/*Calculate the normalized coefficients before returning the P system read*/

	return ENPS;


}


void ENPS_XML_Reader::ENPS_XML_Reader::fillInMembrane(xml_node<>* currentMembrane){
	xml_node<>* currentRegion = currentMembrane->first_node("region");
	fillInRegion(currentRegion);
	xml_node<>* children = currentMembrane->first_node("children");
	/*Read each one of the child membranes*/
	for(xml_node<>* childMembrane = children->first_node("membrane"); childMembrane; childMembrane = childMembrane->next_sibling("membrane"))
		fillInMembrane(childMembrane);

}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInRegion(xml_node<>* currentRegion){
	xml_node<>* memory = currentRegion->first_node("memory");
	xml_node<>* ruleset = memory->next_sibling("rulesList");
	if(read_mode==READ_VARIABLES)
		fillInMemory(memory);
	else if(read_mode=READ_PROGRAMS)
			fillInRulesList(ruleset);



}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInMemory(xml_node<>* memory){
	for (xml_node<> *variable = memory->first_node("variable");
		variable;
		variable = variable->next_sibling("variable"))
		readVariable(variable);
}

void ENPS_XML_Reader::ENPS_XML_Reader::readVariable(xml_node<>* variable){

	xml_node<>* variableName = variable->first_node();
	xml_attribute<>* variableValue = variable->first_attribute("initialValue");
	Results_type variableInitValue = (Results_type)atof(variableValue->value());
	ENPS->variables[variable_counter] = variableInitValue;
	string variableStringName = variableName->value();	
	variableAssociations[variableStringName] = variable_counter;
	variable_counter++;
}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInRulesList(xml_node<>* rulesList){
	for (xml_node<> *rule = rulesList->first_node("rule"); rule; rule = rule->next_sibling("rule")){
		fillInRule(rule);

	}
}

void ENPS_XML_Reader::ENPS_XML_Reader::getVariableNames(char** returnedVariableNames){
	std::map<std::string, Variable_ID>::iterator vaIterator = variableAssociations.begin();
	for(;vaIterator!= variableAssociations.end();++vaIterator){
		strcpy(returnedVariableNames[vaIterator->second], vaIterator->first.c_str());
	}
}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInRule(xml_node<>* rule){
	xml_node<> *repartitionProtocol = rule->first_node("repartitionProtocol");
	xml_node<> *productionFunction = repartitionProtocol->next_sibling("productionFunction");
	xml_node<> *enzyme = productionFunction->next_sibling("enzyme");
	/*Read each part of the rule*/
	fillInRepartitionProtocol(repartitionProtocol);
	fillInProductionFunction(productionFunction);
	readEnzyme(enzyme);
	program_counter++;
}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInRepartitionProtocol(xml_node<>* repartitionProtocol){
	repartition_protocol_counter=0;
	for (xml_node<> *repartitionVariable = repartitionProtocol->first_node("repartitionVariable"); repartitionVariable; repartitionVariable = repartitionVariable->next_sibling("repartitionVariable")){
		fillInRepartitionVariable(repartitionVariable);
	}
}

void ENPS_XML_Reader::ENPS_XML_Reader::fillInProductionFunction(xml_node<>* productionFunction){
	pf_reader.setENPS(this->ENPS);
	pf_reader.setVariableAssociations(this->variableAssociations);
	pf_reader.setProgramCounter(program_counter);
	ENPS->production_function_items[program_counter] = pf_reader.readProductionFunctionElement(productionFunction);
	if(pf_reader.temp_production_function_tree_size>ENPS->max_size_of_production_function_tree)
		ENPS->max_size_of_production_function_tree = pf_reader.temp_production_function_tree_size;
}



void ENPS_XML_Reader::ENPS_XML_Reader::fillInRepartitionVariable(xml_node<>* repartitionVariable){
	std::string variableName = repartitionVariable->first_node()->value();
	xml_attribute<>* variableValue = repartitionVariable->first_attribute("contribution");
	Coefficient_Type coefficient = (Coefficient_Type)atof(variableValue->value());
	Variable_ID variableNumericalID = variableAssociations[variableName];
	ENPS->repartition_protocol_coefficients[program_counter][repartition_protocol_counter] = coefficient;
	ENPS->repartition_protocol_variables[program_counter][repartition_protocol_counter] = variableNumericalID;
	repartition_protocol_counter++;
}


void ENPS_XML_Reader::ENPS_XML_Reader::readEnzyme(xml_node<>* enzyme){

	if(enzyme==0)
		/*If there is no enzyme pointer, set that the program has no enzyme to be checked*/
		ENPS->enzymes[program_counter]=NON_VALID_ENZYME_MASK;
	else{
		std::string enzymeName = enzyme->first_node()->value();
		ENPS->enzymes[program_counter] = variableAssociations[enzymeName];
	}


}


std::map<std::string, Variable_ID> ENPS_XML_Reader::ENPS_XML_Reader::get_variable_associations()
{
	return variableAssociations;

}


