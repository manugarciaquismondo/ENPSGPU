
#include "Results_Writer.h"


ENPS_XML_Reader::Results_Writer::Results_Writer(void)
{

}

ENPS_XML_Reader::Results_Writer::~Results_Writer(void)
{

}


void ENPS_XML_Reader::Results_Writer::writeResults(Results_type* variables, char* fileRoute, float elapsedTime, std::map<std::string, Variable_ID> variable_associations)
{
	this->variables = variables;
	this->variableAssociations = variable_associations;
	write_comment();
	create_root();
	write_variables();
	write_execution_time(elapsedTime);
	write_to_file(fileRoute);

	// root node



}



void ENPS_XML_Reader::Results_Writer::write_comment(){
	xml_node<>* declaration= document.allocate_node(node_declaration);
	declaration->append_attribute(document.allocate_attribute("version", "1.0"));
	declaration->append_attribute(document.allocate_attribute("encoding", "utf-8"));
	document.append_node(declaration);
}

void ENPS_XML_Reader::Results_Writer::create_root(){
	root = document.allocate_node(node_element, "Results");
	root->append_attribute(document.allocate_attribute("version", "1.0"));
	document.append_node(root);

}

void ENPS_XML_Reader::Results_Writer::write_variables(){
	xml_node<>* variablesNode = document.allocate_node(node_element, "Variables");
	write_variable_values(variablesNode);
	root->append_node(variablesNode);

}

void ENPS_XML_Reader::Results_Writer::write_variable_values(xml_node<>* variablesNode){
	std::map<std::string, Variable_ID>::iterator vaIterator = variableAssociations.begin();
	for(;vaIterator!= variableAssociations.end();++vaIterator){
		write_variable(vaIterator->first, vaIterator->second, variablesNode);
	}
}

void ENPS_XML_Reader::Results_Writer::write_variable(std::string variableName, Variable_ID variabeID, xml_node<>* variablesNode){
	transform_to_chars(this->variables[variabeID]);
	xml_node<>* variableNode = document.allocate_node(node_element, "Variable");
	variableNode->append_attribute(document.allocate_attribute("Name", variableName.c_str()));
	variableNode->append_attribute(document.allocate_attribute("Value", buffer));
	variablesNode->append_node(variableNode);
}

void ENPS_XML_Reader::Results_Writer::transform_to_chars(float number){
	
	sprintf_s(buffer, "%.4g", number);
}


void ENPS_XML_Reader::Results_Writer::write_execution_time(float elapsedTime){
	 transform_to_chars(elapsedTime);
	xml_node<>* resultsNode = document.allocate_node(node_element, "ExecutionTime");
	resultsNode->append_attribute(document.allocate_attribute("Value", buffer));
	root->append_node(root);
}

void ENPS_XML_Reader::Results_Writer::write_to_file(char* fileRoute){
	std::string xml_as_string;
	rapidxml::print(std::back_inserter(xml_as_string), document);
	std::ofstream xmlFile;
	xmlFile.open(fileRoute);
	xmlFile << xml_as_string;
	xmlFile.close();
}