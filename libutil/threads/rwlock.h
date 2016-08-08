#ifndef LIBUTIL_RWLOCK_H
#define LIBUTIL_RWLOCK_H

#include "base/rwlock_base.h"

namespace libutil {


/** \brief Read-write lock

    Read-write locks act like mutexes for writing, but allow simultaneous
    access for reading.

    \ingroup libutil_threads
 **/
class rwlock;


} // namespace libutil


#if defined(USE_PTHREADS)

#include "posix/rwlock_posix.h"
namespace libutil {
class rwlock : public rwlock_base<rwlock_posix> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS) && !defined(HAVE_WIN32_SRWLOCK)

#include "windows/rwlock_windows.h"
namespace libutil {
class rwlock : public rwlock_base<rwlock_windows> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS) && defined(HAVE_WIN32_SRWLOCK)

#include "windows/rwlock_windows_srw.h"
namespace libutil {
class rwlock : public rwlock_base<rwlock_windows_srw> { };
} // namespace libutil

#endif


#endif // LIBUTIL_RWLOCK_H
