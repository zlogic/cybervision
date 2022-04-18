#ifndef PTHREAD_H
#define PTHREAD_H

#include <windows.h>

typedef void pthread_mutex_attr;
typedef CRITICAL_SECTION pthread_mutex_t;
typedef void pthread_attr_t;
typedef HANDLE pthread_t;

int pthread_create(pthread_t*, const pthread_attr_t *attr, LPTHREAD_START_ROUTINE, void *arg);
int pthread_join(pthread_t*, void **value_ptr);

int pthread_mutex_init(pthread_mutex_t*, pthread_mutex_attr*);
int pthread_mutex_destroy(pthread_mutex_t*);
int pthread_mutex_lock(pthread_mutex_t*);
int pthread_mutex_unlock(pthread_mutex_t*);

#endif
