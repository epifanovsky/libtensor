#ifndef LIBUTIL_SPINLOCK_H
#define LIBUTIL_SPINLOCK_H

#include "base/mutex_base.h"

namespace libutil {


/** \brief Spinlock

    Spinlock is a mutex that spins in a loop while waiting for the release
    instead of yielding.

    \sa mutex

    \ingroup libutil_threads
 **/
class spinlock;


} // namespace libutil


#if defined(USE_PTHREADS) && defined(HAVE_PTHREADS_SPINLOCK)

#include "posix/spinlock_posix.h"
namespace libutil {
class spinlock : public mutex_base<spinlock_posix> { };
} // namespace libutil


#elif defined(USE_PTHREADS) && !defined(HAVE_PTHREADS_SPINLOCK)

#include "posix/mutex_posix.h"
namespace libutil {
class spinlock : public mutex_base<mutex_posix> { };
} // namespace libutil


#elif defined(USE_PTHREADS) && defined(HAVE_MACOS_SPINLOCK)

#include "macos/spinlock_macos.h"
namespace libutil {
class spinlock : public mutex_base<spinlock_macos> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS)

#include "windows/spinlock_windows.h"
namespace libutil {
class spinlock : public mutex_base<spinlock_windows> { };
} // namespace libutil

#endif


#endif // LIBUTIL_SPINLOCK_H
