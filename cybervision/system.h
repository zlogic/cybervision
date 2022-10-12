#ifndef SYSTEM_H
#define SYSTEM_H

int cpu_cores();
void sleep_ms();
unsigned int thread_id();
#ifdef _MSC_VER
#define strcasecmp _stricmp
#endif

#ifdef WIN32
int rand_r(unsigned int *seedp);
#endif

#endif
