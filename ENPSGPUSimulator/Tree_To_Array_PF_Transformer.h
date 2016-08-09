#ifndef TREE_TO_ARRAY_PF_TRANSFORMER
#define TREE_TO_ARRAY_PF_TRANSFORMER

#include "enps_parameters.h"
#include "ENPS_Model.h"
namespace ENPS_XML_Reader{
	class Tree_To_Array_PF_Transformer
	{
	public:
		Tree_To_Array_PF_Transformer(void);
		~Tree_To_Array_PF_Transformer(void);
		void transform_to_production_function_items_array(ENPS_Model::ENPS_Instance *example);
		Offset_type allocate_item(production_function_item pf_item, ENPS_Model::ENPS_Instance *ENPS, int program);
		void copy_production_function_item(ENPS_Model::ENPS_Instance *ENPS, production_function_item source_copy, int program);
	private:
		Offset_type auxiliary_index;
	};
};

#endif