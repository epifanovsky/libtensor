#include "spinlock_macos.h"

namespace libutil {


void spinlock_macos::create(mutex_id_type &id) {

    id = OS_SPINLOCK_INIT;
}


void spinlock_macos::destroy(mutex_id_type &id) {

    id = OS_SPINLOCK_INIT;
}


void spinlock_macos::lock(mutex_id_type &id) {

    OSSpinLockLock(&id);
}


void spinlock_macos::unlock(mutex_id_type &id) {

    OSSpinLockUnlock(&id);
}


} // namespace libutil
