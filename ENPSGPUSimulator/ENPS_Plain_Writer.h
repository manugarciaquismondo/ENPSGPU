#ifndef ENPS_PLAIN_WRITER
#define ENPS_PLAIN_WRITER
#include <stdio.h>
#include <string>
#include <map>
#include "enps_parameters.h"
#include "ENPS_Model.h"
namespace ENPS_Results_Writer{
class ENPS_Plain_Writer
{
public:
	ENPS_Plain_Writer(void);
	~ENPS_Plain_Writer(void);
	void writeResults(char* fileRoute, Results_type* variables, std::map<std::string, Results_type> variableAssociations);
};
};
#endif