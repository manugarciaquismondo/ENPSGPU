// Archivo DLL principal.


#include "ENPSSequentialSimulator.h"


using namespace ENPS_Model;

ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::ENPSSequentialSimulator(){}
ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::~ENPSSequentialSimulator(){}

int ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::simulate(char* fileName, char* resultsRoute, int cycles, char show_results){


	simulator_show_results = show_results;
	LARGE_INTEGER ticksPerSecond;
	LARGE_INTEGER startTick;   // A point in time
	LARGE_INTEGER endTick;   // A point in time
	QueryPerformanceFrequency(&ticksPerSecond);

	parseENPSFile(fileName);
	allocate_variable_names(number_of_variables);
	if(!this->cycles) this->cycles =cycles;

	QueryPerformanceCounter(&startTick);
    computeENPSSimulation();
	QueryPerformanceCounter(&endTick);

	double timeDifferenceInSeconds = (double)(endTick.QuadPart - startTick.QuadPart)/ (double)ticksPerSecond.QuadPart;
	double timeDifferenceInMilliseconds= timeDifferenceInSeconds*1000.0f;
	if(show_results){
		ENPS_XML_Reader::Results_Printer results_printer;
		results_printer.print_results(example->number_of_variables, example->variables, &reader);
	}
	printf("Total number of cycles: %i. Elapsed time: %4.4f milliseconds\n", cycles, timeDifferenceInMilliseconds);
	reader.getVariableNames(variable_names);
	write_results(number_of_variables, resultsRoute, cycles, numberOfPrograms, timeDifferenceInMilliseconds, example->variables);
	return 0;

}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::parseENPSFile(char* ENPS_route){

	
	example = reader.readENPSFile(ENPS_route);
	ENPS_XML_Reader::Tree_To_Array_PF_Transformer array_pf_transformer;
	array_pf_transformer.transform_to_production_function_items_array(example);
	if(simulator_show_results)
		example->display();
	cycles = reader.getCycles();
	numberOfPrograms= example->number_of_programs;
	number_of_variables= example->number_of_variables;	
	
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::computeENPSSimulation(){
	numberOfPrograms=example->number_of_programs;
	maxSizeOfRepartitionProtocols = example->max_size_of_repartition_protocols;
	initializeDataStructures();
	for(int i=0; i<cycles; i++){
		calculateMinimumForEachProgram();
		checkProgramApplication();
		computeProductionFunctions();
		clearContributedVariables();
		applyRepartitionProtocols();
	}
	freeDataStructures();
}


void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::freeDataStructures(){
	delete [] requiredEnzymes;
	delete [] productionFunctionResults;
	free(applyPrograms);
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::computeProductionFunctions(){
	for(int i=0; i<numberOfPrograms; i++){
		if(applyPrograms[i]==APPLY_PROGRAM)
			productionFunctionResults[i]=computeProductionFunction(example->production_function_items[i]);
	}

}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::applyRepartitionProtocols(){
	for(int programIndex=0; programIndex<numberOfPrograms; programIndex++){
		if(applyPrograms[programIndex]==APPLY_PROGRAM)
			for(int contributionIndex=0; contributionIndex<maxSizeOfRepartitionProtocols; contributionIndex++){
				Coefficient_Type coefficient = example->repartition_protocol_coefficients[programIndex][contributionIndex];
				if(coefficient!=0){
					example->variables[example->repartition_protocol_variables[programIndex][contributionIndex]]+=
					coefficient*productionFunctionResults[programIndex];
				}
			}
	}
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::clearContributedVariables(){
	for(int programIndex = 0; programIndex<numberOfPrograms; programIndex++)
		if(applyPrograms[programIndex]==APPLY_PROGRAM)
			clearContributedVariablesByProgram(example->production_function_items[programIndex]);

}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::clearContributedVariablesByProgram(production_function_item* current_production_function_item){
	Active_marker operation = current_production_function_item->operation;
	if(operation==VARIABLE){
		example->variables[current_production_function_item->variable]=0;
	}
	if((operation|0x1f)==0x1f){
		clearContributedVariablesByProgram(current_production_function_item->left_operand);
		clearContributedVariablesByProgram(current_production_function_item->right_operand);
	}
}



void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::calculateMinimumForEachProgram(){
	int numberOfPrograms = example->number_of_programs;
	
	for(int programIndex=0; programIndex<numberOfPrograms; programIndex++){
		minimum_enzyme_value = MAX_ENZYME_VALUE;
		calculateMimimumEnzymeValue(example->production_function_items[programIndex]);
		requiredEnzymes[programIndex]=minimum_enzyme_value;
		
	}
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::calculateMimimumEnzymeValue(production_function_item* current_production_function_item){
	Enzyme_type local_min_value = MAX_ENZYME_VALUE;
	Active_marker operation = current_production_function_item->operation;
	if(operation==VARIABLE){
		local_min_value = example->variables[current_production_function_item->variable];
	}
	if((operation|0x1f)==0x1f){
		calculateMimimumEnzymeValue(current_production_function_item->left_operand);
		calculateMimimumEnzymeValue(current_production_function_item->right_operand);
	}
	if(local_min_value<minimum_enzyme_value)
		minimum_enzyme_value = local_min_value;

}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::checkProgramApplication(){
	for(int programIndex=0; programIndex<numberOfPrograms; programIndex++){
		Variable_ID enzyme_ID = example->enzymes[programIndex];
		if((enzyme_ID!=NON_VALID_ENZYME_MASK)&&((Enzyme_type)example->variables[enzyme_ID])<=requiredEnzymes[programIndex])
			applyPrograms[programIndex]=NOT_APPLY_PROGRAM;
		else
			applyPrograms[programIndex]=APPLY_PROGRAM;
	
	}
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::initializeDataStructures(){
	requiredEnzymes = new Enzyme_type[numberOfPrograms];
	applyPrograms = (Active_marker*)malloc(example->number_of_programs*sizeof(Active_marker));
	productionFunctionResults= new Results_type[numberOfPrograms];
}

Results_type ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::computeProductionFunction(production_function_item *currentProductionFunctionItem){
	switch(currentProductionFunctionItem->operation){
		case(VARIABLE):
			return example->variables[currentProductionFunctionItem->variable];
			break;
		case(COEFFICIENT):
			return currentProductionFunctionItem->coefficient;
		case(ADD):
			return computeProductionFunction(currentProductionFunctionItem->left_operand)+
					computeProductionFunction(currentProductionFunctionItem->right_operand);
			break;
		case(SUBSTRACT):
			return computeProductionFunction(currentProductionFunctionItem->left_operand)-
					computeProductionFunction(currentProductionFunctionItem->right_operand);
			break;
		case(MULTIPLY):
			return computeProductionFunction(currentProductionFunctionItem->left_operand)*
					computeProductionFunction(currentProductionFunctionItem->right_operand);
			break;
		case(DIV):
			return computeProductionFunction(currentProductionFunctionItem->left_operand)/
					computeProductionFunction(currentProductionFunctionItem->right_operand);
			break;
		case(POW):
			return pow(computeProductionFunction(currentProductionFunctionItem->left_operand),
					(int)computeProductionFunction(currentProductionFunctionItem->right_operand));
			break;
		default:
			return 0;
			break;
	}
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::write_results(int number_of_variables, char* fileRoute, int cycles, int number_of_programs, double elapsedTime, Results_type* simulation_variable_results){
	FILE* resultsFile;
	resultsFile = fopen(fileRoute, "w");
	fprintf(resultsFile, "cycles : 0%i;\n", cycles);
	fprintf(resultsFile, "programs : 0%i;\n", number_of_programs);
	write_variables(resultsFile, number_of_variables, simulation_variable_results);
	fprintf(resultsFile, "time : %6.6f milliseconds;\n", elapsedTime);
	fclose(resultsFile);
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::write_variables(FILE* resultsFile, int number_of_variables, Results_type* simulation_variable_results){
	for(int i=0; i<number_of_variables; i++)
		fprintf(resultsFile, "var : %s : %6.6f ;\n", variable_names[i], simulation_variable_results[i]);
}

void ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator::allocate_variable_names(int number_of_variables){
	variable_names=(char**)malloc(sizeof(char*)*number_of_variables);
	for(int i=0; i<number_of_variables; i++)
		variable_names[i]=(char*)malloc(sizeof(char)*256);
}