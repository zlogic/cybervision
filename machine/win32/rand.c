#include "rand.h"
#include <stdlib.h>

int rand_r(unsigned int *seedp)
{
    return (int)rand();
}
