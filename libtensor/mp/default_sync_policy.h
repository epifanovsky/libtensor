#ifndef LIBTENSOR_DEFAULT_SYNC_POLICY_H
#define LIBTENSOR_DEFAULT_SYNC_POLICY_H

#include <libutil/threads/mutex.h>
#include <libutil/threads/rwlock.h>

namespace libtensor {


/** \brief Default synchronization policy

    \ingroup libtensor_mp
 **/
class default_sync_policy {
public:
    typedef libutil::mutex mutex_t; //!< Mutex type
    typedef libutil::rwlock rwlock_t; //!< Read-write lock type

};


} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYNC_POLICY_H
