

#ifndef ENPS_XML_READER
#define ENPS_XML_READER

#include "../rapidxml-1.13/rapidxml.hpp"
#include <map>
#include <string>
#include <iostream>
#include <fstream>
#include "ENPS_Model.h"
#include "ENPS_XML_Production_function_reader.h"

using namespace rapidxml;


	
	using namespace rapidxml;
	namespace ENPS_XML_Reader{

		class ENPS_XML_Reader
		{
		public:
			ENPS_XML_Reader();
			~ENPS_XML_Reader();

		


			ENPS_Model::ENPS_Instance* ENPS;
			ENPS_Model::ENPS_Instance* readENPSFile(char* route);
			std::map<std::string, Variable_ID> get_variable_associations();
			long getCycles();
			void getVariableNames(char** returnedVariableNames);
			

		private:
			void readCycles(xml_node<>* cyclesNode);
			long cycles;
			unsigned char read_mode;
			ENPS_XML_Reader::ENPS_XML_Production_function_reader pf_reader;
			std::map<std::string, Variable_ID> variableAssociations;
			void readENPSDescription(xml_document<> *source);
			void fillInParameters(xml_node<> *parameters);
			void fillInRegion(xml_node<> *currentRegion);
			void fillInMemory(xml_node<> *memory);
			void fillInVariable(xml_node<> *variable);
			void fillInRulesList(xml_node<> *rulesList);
			void fillInRule(xml_node<> *rule);
			void fillInRepartitionProtocol(xml_node<> *repartitionProtocol);

			void fillInMembrane(xml_node<>* currentMembrane);
			void fillInRepartitionVariable(xml_node<>* repartitionVariable);
			void fillInProductionFunction(xml_node<>* productionFunction);
			void readVariable(xml_node<>* variable);
			void readEnzyme(xml_node<>* enzyme);
	


			int repartition_protocol_counter;

			int variable_counter;
			int program_counter;


			
		};

	};
#endif
