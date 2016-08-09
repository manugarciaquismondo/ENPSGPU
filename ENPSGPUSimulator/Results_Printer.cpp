
#include "Results_Printer.h"
#include <stdio.h>

using namespace std;

ENPS_XML_Reader::Results_Printer::Results_Printer(void)
{
}

ENPS_XML_Reader::Results_Printer::~Results_Printer(void)
{
}

void ENPS_XML_Reader::Results_Printer::print_results(int number_of_variables, Results_type* resulting_variable_values, ENPS_XML_Reader *reader){

	reverse_variable_associations(reader);
	for(int i=0; i<number_of_variables; i++){
		fprintf(stdout, "Variable: %s, Index: %i, Value: %4.2f\n", reverseVariableAssociations[i].c_str(), i, resulting_variable_values[i]);
	}
}


void ENPS_XML_Reader::Results_Printer::reverse_variable_associations(ENPS_XML_Reader *reader){
	map<string, Variable_ID> variableAssociations = reader->get_variable_associations();
	map<string, Variable_ID>::iterator vaIterator = variableAssociations.begin();
	for(;vaIterator!= variableAssociations.end();++vaIterator){
		reverseVariableAssociations[vaIterator->second] = vaIterator->first;
	}

}