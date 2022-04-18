#include "pthread.h"
               
int pthread_create(pthread_t* thread, const pthread_attr_t *attr, LPTHREAD_START_ROUTINE start_routine, void *arg)
{
    *thread = CreateThread(NULL, 0, start_routine, arg, 0, NULL);
    return *thread != NULL;
}

int pthread_join(pthread_t* thread, void **value_ptr)
{
    WaitForSingleObject(thread, INFINITE);
    CloseHandle(thread);
    return 0;
}

int pthread_mutex_init(pthread_mutex_t* mutex, pthread_mutex_attr* attr)
{
    InitializeCriticalSection(mutex);
    return 0;
}

int pthread_mutex_destroy(pthread_mutex_t* mutex)
{
    DeleteCriticalSection(mutex);
    return 0;
}

int pthread_mutex_lock(pthread_mutex_t* mutex)
{
    EnterCriticalSection(mutex);
    return 0;
}

int pthread_mutex_unlock(pthread_mutex_t* mutex)
{
    LeaveCriticalSection(mutex);
    return 0;
}
