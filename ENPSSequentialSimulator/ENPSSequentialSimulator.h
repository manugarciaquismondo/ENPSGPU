// ENPSSequentialSimulator.h

#ifndef ENPS_SEQUENTIAL_SIMULATOR
#define ENPS_SEQUENTIAL_SIMULATOR
#include "../ENPSGPUSimulator/ENPS_XML_Reader.h"
#include "../ENPSGPUSimulator/Tree_To_Array_PF_Transformer.h"
#include "../ENPSGPUSimulator/Results_Printer.h"
#include "../ENPSGPUSimulator/ENPS_Model.h"
#include "../ENPSGPUSimulator/enps_parameters.h"
#include <cmath>
#include <Windows.h>
#include <iomanip>
#include <float.h>

#define MAX_ENZYME_VALUE FLT_MAX

namespace ENPSSequentialSimulatorNamespace {

	
	public class ENPSSequentialSimulator
	{
	public:
		ENPSSequentialSimulator();
		~ENPSSequentialSimulator();
		int simulate(char* fileName, char* resultsRoute, int cycles, char show_results);

	private:
		ENPS_Model::ENPS_Instance *example;
		int cycles;
		char simulator_show_results;
		int number_of_variables;
		char** variable_names;
		int numberOfPrograms;
		int maxSizeOfRepartitionProtocols;
		Enzyme_type *requiredEnzymes;
		Results_type *productionFunctionResults;
		Enzyme_type minimum_enzyme_value;
		Active_marker *applyPrograms;
		ENPS_XML_Reader::ENPS_XML_Reader reader;
		Results_type computeProductionFunction(production_function_item* currentProductionFunctionItem);
		void parseENPSFile(char* ENPS_route);
		void computeENPSSimulation();
		void computeProductionFunctions();
		void calculateMinimumForEachProgram();
		void clearContributedVariables();
		void clearContributedVariablesByProgram(production_function_item* current_production_function_item);
		void applyRepartitionProtocols();
		void freeDataStructures();
		void write_results(int number_of_variables, char* resultsRoute, int cycles, int number_of_programs, double elapsedTime, Results_type* simulatio_variable_results);
		void write_variables(FILE* resultsFile, int number_of_variables, Results_type* simulation_variable_results);
		void allocate_variable_names(int number_of_variables);
		void calculateMimimumEnzymeValue(production_function_item* current_production_function_item);
		void checkProgramApplication();
		void initializeDataStructures();


		// TODO: agregar aquí los métodos de la clase.
	};
}
#endif