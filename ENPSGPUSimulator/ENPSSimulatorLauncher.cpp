#pragma once
#include "ENPSSimulatorLauncher.h"

ENPSSimulatorLauncher::ENPSSimulatorLauncher(void)
{
}

ENPSSimulatorLauncher::~ENPSSimulatorLauncher(void)
{
}

int ENPSSimulatorLauncher::callENPSSimulator(int argc, char **argv)
{
	simulateENPS(argv[1], argv[3], atoi(argv[2]), atoi(argv[4]));
	return 0;
}
