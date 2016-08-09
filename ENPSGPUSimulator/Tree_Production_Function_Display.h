
#ifndef TREE_PRODUCTION_FUNCTION_DISPLAY
#define TREE_PRODUCTION_FUNCTION_DISPLAY
#include "ENPS_Model.h"

#include <string>


namespace ENPS_Model{
	class Tree_Production_Function_Display
	{
	public:
		Tree_Production_Function_Display(void);
		~Tree_Production_Function_Display(void);
		void set_test_class(ENPS_Model::ENPS_Instance *ENPS);

		void display_production_function(int i);
		void arrayDisplay(int i);
	private:
		ENPS_Model::ENPS_Instance *ENPS;
		std::string array_display_production_function(int program, Offset_type element);
		std::string display_production_function_item(production_function_item *pf_item);
	};
};

#endif