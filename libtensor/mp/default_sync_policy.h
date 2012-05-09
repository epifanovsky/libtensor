#ifndef LIBTENSOR_DEFAULT_SYNC_POLICY_H
#define LIBTENSOR_DEFAULT_SYNC_POLICY_H

#include "threads.h"

namespace libtensor {


/** \brief Default synchronization policy

    \ingroup libtensor_mp
 **/
class default_sync_policy {
public:
    typedef mutex mutex_t; //!< Mutex type
    typedef rwlock rwlock_t; //!< Read-write lock type

};


} // namespace libtensor

#endif // LIBTENSOR_DEFAULT_SYNC_POLICY_H
