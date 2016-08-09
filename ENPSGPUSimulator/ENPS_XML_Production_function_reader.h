#ifndef ENPS_XML_PF_READER
#define ENPS_XML_PF_READER

#include "ENPS_Model.h"
#include <map>
#include <string>
#include "../rapidxml-1.13/rapidxml.hpp"


using namespace rapidxml;

namespace ENPS_XML_Reader{

	class ENPS_XML_Production_function_reader
	{
	public:
		ENPS_XML_Production_function_reader(void);
		~ENPS_XML_Production_function_reader(void);
		//void fillInProductionFunction(xml_node<> *productionFunction);
		void setVariableAssociations(std::map<std::string, Variable_ID> variableAssociations);
		void setENPS(ENPS_Model::ENPS_Instance *ENPS);
		void setProgramCounter(int program_counter);
		production_function_item* readProductionFunctionElement(xml_node<>* treeElement);
		int temp_production_function_tree_size;
	private:
		int program_counter;
		ENPS_Model::ENPS_Instance *ENPS;
		std::map<std::string, Variable_ID> variableAssociations;
		int production_function_counter;
		xml_node<>* common_production_function_variable;
		void readProductionFunctionVariable(xml_node<>* first_pf_element);
		void readProductionFunctionCoefficient(xml_node<>* coefficient_element);
		production_function_item* readTreeElement(xml_node<> *treeElement);

	};

};

#endif