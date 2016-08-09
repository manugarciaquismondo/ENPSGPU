
#include "kernel.cuh"

Coefficient_Type* flattened_pf_coefficients;
Variable_ID* flattened_pf_variables;
Active_marker* flattened_pf_operations;
Offset_type* flattened_pf_left_child_offsets;
Offset_type* flattened_pf_right_child_offsets;
Coefficient_Type* flattened_rp_coefficients;
Variable_ID* flattened_rp_variables;
char** variable_names;

	/*Declare pointers for Memcpy operations. These pointers will contain info located in the instance passed as argument*/
Results_type *dev_variables;
Variable_ID *dev_enzymes;
Coefficient_Type *dev_production_function_coefficients;
Variable_ID *dev_production_function_variables;
production_function_item* dev_production_function_items;
Coefficient_Type *dev_repartition_protocol_coefficients;
Variable_ID *dev_repartition_protocol_variables = 0;
Active_marker* dev_production_function_operations;
Offset_type *dev_production_function_left_child_offset;
Offset_type *dev_production_function_right_child_offset;

	/*Declare pointers only for Malloc operations. These pointers will contain the results of operations on GPU, but they do not reference data originally located at the host*/
Results_type* dev_production_function_results;
Enzyme_type *dev_required_enzymes;
Active_marker *dev_apply_programs;

Results_type *simulation_variable_results;
cudaError_t cudaStatus;
char simulator_show_results;
cudaEvent_t start, stop;



extern "C" int simulateENPS(char* simulatorRoute, char* resultsRoute, int iterations, char show_results)
{
	simulator_show_results = show_results;
	ENPS_Instance *example;
	int cycles;
	ENPS_XML_Reader::ENPS_XML_Reader reader;
	example = reader.readENPSFile(simulatorRoute);
	cycles = reader.getCycles();
	ENPS_XML_Reader::Tree_To_Array_PF_Transformer array_pf_transformer;
	array_pf_transformer.transform_to_production_function_items_array(example);
	if(simulator_show_results)
		example->display();
	flattenElements(example);
	int number_of_programs = example->number_of_programs;
	int number_of_variables= example->number_of_variables;
	allocate_variable_names(number_of_variables);

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if(!cycles) cycles = iterations;

    cudaError_t cudaStatus = computeENPSsimulation(example, cycles);
	float elapsedTime;
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }
	


	// cudaThreadExit must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaThreadExit();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaThreadExit failed!");
        return 1;
    }

	
	if(simulator_show_results){
		ENPS_XML_Reader::Results_Printer results_printer;	
		results_printer.print_results(number_of_variables, simulation_variable_results, &reader);
	}
	printf("Total number of cycles: %i. Elapsed time: %4.4f milliseconds\n", cycles, elapsedTime);
	reader.getVariableNames(variable_names);
	write_results(number_of_variables, resultsRoute, cycles, number_of_programs, elapsedTime);
	freeHostPointers();

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t computeENPSsimulation(ENPS_Instance* example, int iterations)
{


	if(simulator_show_results)
		example->display();
	



    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }





	if (performHostDataMemallocs(example)) goto Error;
	if (performHostDataMemcpys(example)) goto Error;
	if (performDeviceDataMemallocs(example)) goto Error;

	


	

/*Once the setups steps are completed, go on with the simulation steps as such*/
	cudaEventRecord(start);
	for(int i=0; i<iterations; i++){
		//setRequiredEnzymesAtMaximumValue<<<1, example->number_of_programs>>>(dev_required_enzymes);

		calculate_minimum_for_each_program<<<example->number_of_programs/BLOCK_SIZE+1, BLOCK_SIZE>>>(dev_production_function_operations, dev_enzymes, dev_production_function_variables, dev_required_enzymes, dev_variables, example->max_size_of_production_functions, example->number_of_programs);

		clear_production_function_results<<<example->number_of_programs/BLOCK_SIZE+1, BLOCK_SIZE>>>(dev_production_function_results, dev_apply_programs, dev_enzymes, dev_variables, dev_required_enzymes, example->number_of_programs);

		compute_production_function<<<example->number_of_programs/BLOCK_SIZE+1, BLOCK_SIZE>>>(dev_production_function_left_child_offset, dev_production_function_right_child_offset, dev_production_function_coefficients, dev_production_function_variables, dev_variables, dev_production_function_results, dev_apply_programs,dev_production_function_operations, example->max_size_of_production_functions, example->number_of_programs);

		clear_contributed_variables<<<example->number_of_programs, example->max_size_of_production_functions>>>(dev_production_function_operations, dev_production_function_variables, dev_variables, dev_apply_programs);

		apply_repartition_protocol<<<example->number_of_programs, example->max_size_of_repartition_protocols>>>(dev_production_function_results, dev_repartition_protocol_coefficients, dev_repartition_protocol_variables, dev_variables, dev_apply_programs);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	if(copySimulationResultsToHostMemory(example->number_of_variables)) goto Error;

Error:

	/*Delete the problem instance*/
	delete example;

	freeGPUPointers();


    
    return cudaStatus;
}


void flattenElements(ENPS_Instance* example){
	int number_of_programs = example->number_of_programs;
	
	/*Allocate memory for the flattened arrays*/

	int i,j;
	flattened_pf_coefficients =(Coefficient_Type*)malloc(example->max_number_of_production_function_items*sizeof(Coefficient_Type));
	flattened_pf_variables = (Variable_ID*)malloc(example->max_number_of_production_function_items*sizeof(Variable_ID));
	flattened_pf_operations = (Active_marker*)malloc(example->max_number_of_production_function_items*sizeof(Active_marker));
	flattened_pf_left_child_offsets =(Offset_type*)malloc(example->max_number_of_production_function_items*sizeof(Offset_type));
	flattened_pf_right_child_offsets = (Offset_type*)malloc(example->max_number_of_production_function_items*sizeof(Offset_type));


	flattened_rp_coefficients = (Coefficient_Type*)malloc(example->max_number_of_repartition_protocol_items*sizeof(Coefficient_Type));
	flattened_rp_variables = (Variable_ID*)malloc(example->max_number_of_repartition_protocol_items*sizeof(Variable_ID));

	for(i=0; i<number_of_programs; i++){
		for(j=0; j<example->max_size_of_production_functions; j++){
			int program_offset = i*example->max_size_of_production_functions;
			flattened_pf_coefficients[program_offset+j] = example->production_function_coefficients[i][j];
			flattened_pf_variables[program_offset+j] = example->production_function_variables[i][j];
			flattened_pf_operations[program_offset+j] = example->production_function_operations[i][j];
			flattened_pf_left_child_offsets[program_offset+j] = example->left_child_offset[i][j];
			flattened_pf_right_child_offsets[program_offset+j] = example->right_child_offset[i][j];
		}
		for(j=0; j<example->max_size_of_repartition_protocols; j++){
			int program_offset = i*example->max_size_of_repartition_protocols;
			flattened_rp_coefficients[program_offset+j] = example->repartition_protocol_coefficients[i][j];
			flattened_rp_variables[program_offset+j] = example->repartition_protocol_variables[i][j];
		}
	}
}

int performHostDataMemallocs(ENPS_Instance *example){

	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_enzymes, example->number_of_programs * sizeof(Variable_ID))/*, "enzymes"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_coefficients, example->max_number_of_production_function_items * sizeof(Coefficient_Type))/*, "production function coefficients"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_variables, example->max_number_of_production_function_items *sizeof(Variable_ID))/*, "production function variables"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_operations, example->max_number_of_production_function_items *sizeof(Active_marker))/*, "production function operations"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_left_child_offset, example->max_number_of_production_function_items *sizeof(Offset_type))/*, "production function left child offset"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_right_child_offset, example->max_number_of_production_function_items *sizeof(Offset_type))/*, "production function right child offset"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_variables, example->number_of_variables*sizeof(Results_type))/*, "variables"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_repartition_protocol_coefficients, example->max_number_of_repartition_protocol_items*sizeof(Coefficient_Type))/*, "repartition protocol coefficients"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_repartition_protocol_variables, example->max_number_of_repartition_protocol_items*sizeof(Variable_ID))/*, "repartition protocol variables"*/)) return -1;
	return 0;
}


int performHostDataMemcpys(ENPS_Instance *example){

	/*These Memcpy copy the data contained in the host to pointers within the GPU. Note that these GPU pointers have already been Malloced*/
	cudaStatus = cudaMemcpy(dev_enzymes, example->enzymes, example->number_of_programs * sizeof(Variable_ID), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for enzymes failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_production_function_coefficients, flattened_pf_coefficients, example->max_number_of_production_function_items * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for production function coefficients failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_production_function_variables, flattened_pf_variables, example->max_number_of_production_function_items * sizeof(Variable_ID), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for production function variables failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_variables, example->variables, example->number_of_variables * sizeof(Results_type), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for variables failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_production_function_operations, flattened_pf_operations, example->max_number_of_production_function_items * sizeof(Active_marker), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for production function operations failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_production_function_left_child_offset, flattened_pf_left_child_offsets, example->max_number_of_production_function_items * sizeof(Offset_type), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for left child offsets failed!");
         return -1;
    }

		cudaStatus = cudaMemcpy(dev_production_function_right_child_offset, flattened_pf_right_child_offsets, example->max_number_of_production_function_items * sizeof(Offset_type), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for right child offsets failed!");
         return -1;
    }


	
	cudaStatus = cudaMemcpy(dev_repartition_protocol_coefficients, flattened_rp_coefficients, example->max_number_of_repartition_protocol_items * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for repartition protocol coefficients failed!");
         return -1;
    }

	cudaStatus = cudaMemcpy(dev_repartition_protocol_variables, flattened_rp_variables, example->max_number_of_repartition_protocol_items * sizeof(Variable_ID), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy for repartition protocol variables failed!");
         return -1;
    }
	return 0;
}

int performDeviceDataMemallocs(ENPS_Instance *example){

	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_production_function_results, example->number_of_programs * sizeof(Results_type))/*, "production function results"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_apply_programs, example->number_of_programs * sizeof(Active_marker))/*, "apply programs"*/)) return -1;
	if(perform_cuda_malloc_and_check_error(cudaMalloc((void**)&dev_required_enzymes, example->number_of_programs * sizeof(Enzyme_type))/*, "required enzymes"*/)) return -1;
	return 0;
}


int perform_cuda_malloc_and_check_error(cudaError_t status){
	/* = "cudaMalloc for ";
	full_error_message = full_error_message.append(error_message);
	full_error_message = full_error_message.append(" failed!");*/
	if(status!= cudaSuccess){
		fprintf(stderr, "cudaMalloc failed!");
		return -1;
	}
	return 0;
}

int copySimulationResultsToHostMemory(int number_of_variables){
	simulation_variable_results = (Results_type*)malloc(number_of_variables*sizeof(Results_type));
	cudaStatus = cudaMemcpy(simulation_variable_results, dev_variables, number_of_variables*sizeof(Results_type), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {		
        fprintf(stderr, "cudaMemcpy for simulation results failed!");		
		return -1;
    }
	return 0;


}

void write_results(int number_of_variables, char* fileRoute, int cycles, int number_of_programs, float elapsedTime){
	FILE* resultsFile;
	resultsFile = fopen(fileRoute, "w");
	fprintf(resultsFile, "cycles : 0%i;\n", cycles);
	fprintf(resultsFile, "programs : 0%i;\n", number_of_programs);
	write_variables(resultsFile, number_of_variables);
	fprintf(resultsFile, "time : %6.6f milliseconds;\n", elapsedTime);
	fclose(resultsFile);
}

void write_variables(FILE* resultsFile, int number_of_variables){
	for(int i=0; i<number_of_variables; i++)
		fprintf(resultsFile, "var : %s : %6.6f ;\n", variable_names[i], simulation_variable_results[i]);
}

void allocate_variable_names(int number_of_variables){
	variable_names=(char**)malloc(sizeof(char*)*number_of_variables);
	for(int i=0; i<number_of_variables; i++)
		variable_names[i]=(char*)malloc(sizeof(char)*256);
}

void freeGPUPointers(){
		/*Free the GPU pointers to data originally contained within the host*/
	cudaFree(dev_variables);
	cudaFree(dev_enzymes);
	cudaFree(dev_production_function_coefficients);
	cudaFree(dev_production_function_variables);
	cudaFree(dev_production_function_operations);
	cudaFree(dev_production_function_left_child_offset);
	cudaFree(dev_production_function_right_child_offset);
	cudaFree(dev_repartition_protocol_coefficients);
	cudaFree(dev_repartition_protocol_variables);


	/*Free the GPU pointers to data used to store GPU calculations*/
	cudaFree(dev_production_function_results);
	cudaFree(dev_required_enzymes);
	cudaFree(dev_apply_programs);
}

void freeHostPointers(){
	free(flattened_pf_coefficients);
	free(flattened_pf_variables);
	free(flattened_pf_operations);
	free(flattened_pf_left_child_offsets);
	free(flattened_pf_right_child_offsets);
	free(flattened_rp_coefficients);
	free(flattened_rp_variables);
	free(simulation_variable_results);
}