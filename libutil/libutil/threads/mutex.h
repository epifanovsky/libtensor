#ifndef LIBUTIL_MUTEX_H
#define LIBUTIL_MUTEX_H

#include "base/mutex_base.h"

namespace libutil {


/** \brief Mutex

    Mutexes prevents simultaneous access of two threads to a protected object.

    \ingroup libutil_threads
 **/
class mutex;


} // namespace libutil


#if defined(USE_PTHREADS)

#include "posix/mutex_posix.h"
namespace libutil {
class mutex : public mutex_base<mutex_posix> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS)

#include "windows/mutex_windows.h"
namespace libutil {
class mutex : public mutex_base<mutex_windows> { };
} // namespace libutil

#endif


#endif // LIBUTIL_MUTEX_H
