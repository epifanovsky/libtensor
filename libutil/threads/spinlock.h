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

#ifdef __APPLE__

#include "macos/spinlock_macos.h"
namespace libutil {
class spinlock : public mutex_base<spinlock_macos> { };
} // namespace libutil

#else

#include "posix/spinlock_posix.h"
namespace libutil {
class spinlock : public mutex_base<spinlock_posix> { };
} // namespace libutil

#endif

#endif // LIBUTIL_SPINLOCK_H
