#ifndef ENPS_SIMULATOR_LAUNCHER
#define ENPS_SIMULATOR_LAUNCHER
#include "kernel.cuh"
class ENPSSimulatorLauncher
{
public:
	ENPSSimulatorLauncher(void);
	~ENPSSimulatorLauncher(void);
	int callENPSSimulator(int argc, char** argv);
};
#endif