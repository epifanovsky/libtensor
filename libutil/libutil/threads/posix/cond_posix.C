#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 500
#endif // _XOPEN_SOURCE
#include <libutil/exceptions/util_exceptions.h>
#include "cond_posix.h"

namespace libutil {


void cond_posix::create(cond_id_type &id) {

    int init = 0;

#ifdef HAVE_PTHREADS_ADAPTIVE_MUTEX
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
    init = pthread_mutex_init(&id.m_mtx, &attr);
    pthread_mutexattr_destroy(&attr);
#else
    init = pthread_mutex_init(&id.m_mtx, 0);
#endif // HAVE_PTHREADS_ADAPTIVE_MUTEX

    if(init) {
        throw threads_exception("cond_posix", "create(cond_id_type &)",
                __FILE__, __LINE__, "Error initializing a mutex.");
    }
    if(pthread_cond_init(&id.m_cond, 0)) {
        throw threads_exception("cond_posix", "create(cond_id_type &)",
                __FILE__, __LINE__, "Error initializing a condition.");
    }
    id.m_sig = false;
}


void cond_posix::destroy(cond_id_type &id) {

    pthread_mutex_destroy(&id.m_mtx);
    pthread_cond_destroy(&id.m_cond);
}


void cond_posix::wait(cond_id_type &id) {

    pthread_mutex_lock(&id.m_mtx);
    if(!id.m_sig) {
        pthread_cond_wait(&id.m_cond, &id.m_mtx);
    }
    id.m_sig = false;
    pthread_mutex_unlock(&id.m_mtx);
}


void cond_posix::signal(cond_id_type &id) {

    pthread_mutex_lock(&id.m_mtx);
    if(!id.m_sig) {
        id.m_sig = true;
        pthread_cond_signal(&id.m_cond);
    }
    pthread_mutex_unlock(&id.m_mtx);
}


void cond_posix::broadcast(cond_id_type &id) {

    pthread_mutex_lock(&id.m_mtx);
    if(!id.m_sig) {
        id.m_sig = true;
        pthread_cond_broadcast(&id.m_cond);
    }
    pthread_mutex_unlock(&id.m_mtx);
}


} // namespace libutil
