#include "rwlock_posix.h"

namespace libutil {


void rwlock_posix::create(rwlock_id_type &id) {

    pthread_rwlock_init(&id, 0);
}


void rwlock_posix::destroy(rwlock_id_type &id) {

    pthread_rwlock_destroy(&id);
}


void rwlock_posix::rdlock(rwlock_id_type &id) {

    pthread_rwlock_rdlock(&id);
}


void rwlock_posix::wrlock(rwlock_id_type &id) {

    pthread_rwlock_wrlock(&id);
}


void rwlock_posix::unlock(rwlock_id_type &id) {

    pthread_rwlock_unlock(&id);
}


} // namespace libutil
