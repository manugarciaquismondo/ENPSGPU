#ifndef RESULTS_WRITER
#define RESULTS_WRITER

#include "../rapidxml-1.13/rapidxml.hpp"
#include "../rapidxml-1.13/rapidxml_print.hpp"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include "ENPS_Model.h"


using namespace rapidxml;

namespace ENPS_XML_Reader{
class Results_Writer
{
public:
	Results_Writer(void);
	~Results_Writer(void);
	void writeResults(Results_type* variables, char* fileRoute, float elapsedTime, std::map<std::string, Variable_ID> variable_associations);
private:
	char buffer[256];
	rapidxml::xml_document<> document;
	rapidxml::xml_node<>* root;
	Results_type* variables;
	std::map<std::string, Variable_ID> variableAssociations;
protected:
	void create_root();
	void write_comment();
	void write_variables();
	void write_variable_values(rapidxml::xml_node<>* variablesNode);
	void write_variable(std::string variableName, Variable_ID variabeID, rapidxml::xml_node<>* variablesNode);
	void write_execution_time(float elapsedTime);
	void write_to_file(char* fileRoute);
	void transform_to_chars(float number);
};
};
#endif