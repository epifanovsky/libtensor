#include "mutex_posix.h"

namespace libutil {


void mutex_posix::create(mutex_id_type &id) {

#ifdef HAVE_PTHREADS_ADAPTIVE_MUTEX
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ADAPTIVE_NP);
    pthread_mutex_init(&id, &attr);
    pthread_mutexattr_destroy(&attr);

#else
    pthread_mutex_init(&id, 0);
#endif // HAVE_PTHREADS_ADAPTIVE_MUTEX
}


void mutex_posix::destroy(mutex_id_type &id) {

    pthread_mutex_destroy(&id);
}


void mutex_posix::lock(mutex_id_type &id) {

    pthread_mutex_lock(&id);
}


void mutex_posix::unlock(mutex_id_type &id) {

    pthread_mutex_unlock(&id);
}


} // namespace libutil
