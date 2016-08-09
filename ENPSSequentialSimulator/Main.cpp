#pragma once

#include "ENPSSequentialSimulator.h"

using namespace std;

int main(int argc, char** argv){
	ENPSSequentialSimulatorNamespace::ENPSSequentialSimulator simulator;
	simulator.simulate(argv[1], argv[3], atoi(argv[2]), atoi(argv[4]));
	

}