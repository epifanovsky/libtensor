#ifndef LIBUTIL_RWLOCK_H
#define LIBUTIL_RWLOCK_H

#include "base/rwlock_base.h"
#include "posix/rwlock_posix.h"

namespace libutil {


/** \brief Read-write lock

    Read-write locks act like mutexes for writing, but allow simultaneous
    access for reading.

    \ingroup libutil_threads
 **/
class rwlock : public rwlock_base<rwlock_posix> { };


} // namespace libutil

#endif // LIBUTIL_RWLOCK_H
