#include "spinlock_posix.h"

namespace libutil {


void spinlock_posix::create(mutex_id_type &id) {

    pthread_spin_init(&id, 0);
}


void spinlock_posix::destroy(mutex_id_type &id) {

    pthread_spin_destroy(&id);
}


void spinlock_posix::lock(mutex_id_type &id) {

    pthread_spin_lock(&id);
}


void spinlock_posix::unlock(mutex_id_type &id) {

    pthread_spin_unlock(&id);
}


} // namespace libutil
