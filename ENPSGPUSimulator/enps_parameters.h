#ifndef ENPS_PARAMETERS
#define ENPS_PARAMETERS
#include <float.h>
#include <limits.h>
#define NUMBER_OF_RULES 10
#define NON_VALID_ENZYME_MASK UINT_MAX
#define NON_VALID_VARIABLE UINT_MAX
#define NON_VALID_COEFFICIENT UINT_MAX >> 1
#define ENZYME_MAX_VALUE FLT_MAX
#define READ_VARIABLES 0x00
#define READ_PROGRAMS 0x01
#define NOT_APPLY_PROGRAM 0X00

#define APPLY_PROGRAM 0X01
#define NO_OPERATION 0X00
#define SUBSTRACT 0x01
#define MULTIPLY 0x02
#define MULTIPLY_AND_SUBSTRACT 0x03
#define ADD 0x04
#define DIV 0x08
#define POW 0x10
#define COEFFICIENT 0x20
#define VARIABLE 0x40
typedef unsigned short Membrane_Label;
typedef unsigned int Variable_ID;
typedef unsigned char Active_marker;
typedef unsigned char Offset_type;


/*We define special types to be changed depending on the GPU version to be executed*/

typedef float Coefficient_Type;
typedef float Results_type;
typedef float Enzyme_type;




#endif