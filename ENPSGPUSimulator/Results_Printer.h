
#ifndef ENPS_RESULTS_PRINTER
#define ENPS_RESULTS_PRINTER
#include "enps_parameters.h"
#include "ENPS_XML_Reader.h"
#include <map>
#include <string>

namespace ENPS_XML_Reader{

	class Results_Printer
	{
	public:
		Results_Printer(void);
		~Results_Printer(void);
		void print_results(int number_of_variables, Results_type* resulting_variable_values, ENPS_XML_Reader *reader);
	private:
		std::map<Variable_ID, std::string> reverseVariableAssociations;
		void reverse_variable_associations(ENPS_XML_Reader *reader);
	};
};

#endif
