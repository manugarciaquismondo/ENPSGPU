// Test_Cases_Library.h

#ifndef TEST_CASES_LIBRARY
#define TEST_CASES_LIBRARY


#include "enps_parameters.h"
#include "production_function_item.h"




namespace ENPS_Model {

/*#define NUMBER_OF_MEMBRANES 2
#define NUMBER_OF_VARIABLES 3
#define TOTAL_NUMBER_OF_VARIABLES NUMBER_OF_MEMBRANES * NUMBER_OF_VARIABLES
#define NUMBER_OF_PROGRAMS 1
#define MAX_SIZE_OF_PRODUCTION_FUNCTIONS 1
#define MAX_SIZE_OF_REPARTITION_PROTOCOLS 2
#define TOTAL_NUMBER_OF_PROGRAMS NUMBER_OF_MEMBRANES * NUMBER_OF_PROGRAMS
#define MAX_NUMBER_OF_PRODUCTION_FUNCTION_ITEMS TOTAL_NUMBER_OF_PROGRAMS * MAX_SIZE_OF_PRODUCTION_FUNCTIONS
#define MAX_NUMBER_OF_REPARTITION_PROTOCOL_ITEMS TOTAL_NUMBER_OF_PROGRAMS * MAX_SIZE_OF_REPARTITION_PROTOCOLS*/

	class ENPS_Instance
	{
	public:

		ENPS_Instance();
		~ENPS_Instance();


		int number_of_variables;
		int number_of_programs;
		int max_size_of_production_functions;
		int max_size_of_repartition_protocols;
		int max_number_of_production_function_items;
		int max_number_of_repartition_protocol_items;
		int max_size_of_production_function_tree;

		void calculateMaxSizes();
		void display();
		
		production_function_item** production_function_items;


		Results_type* variables;
		Variable_ID* enzymes;

		Coefficient_Type** production_function_coefficients;
		Coefficient_Type** repartition_protocol_coefficients;
		Variable_ID** production_function_variables;
		Variable_ID** repartition_protocol_variables;
		Offset_type** left_child_offset;
		Offset_type** right_child_offset;
		Active_marker** production_function_operations;
		void normalizeCoefficients();
		void free_production_function_item(production_function_item* item);

	private:

		void print_constants();
		void initializeDataStructures();
		void clearValues();
		//void print_according_to_operation(int i);
		void print_enzyme(int index);
		void free_pointer_structures();
		int is_operation(Active_marker operation);
		int is_valid_flag(Active_marker operation);


	};
};
#endif