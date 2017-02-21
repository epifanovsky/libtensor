#ifndef LIBUTIL_COND_H
#define LIBUTIL_COND_H

#include "base/cond_base.h"

namespace libutil {


/** \brief Conditional variable

    The conditional variable suspends the current thread until the underlying
    condition is met. This tool enables one thread to wait for a signal from
    another thread.

    There are two states: on and off. The conditional is reusable.

    Three methods are provided:
     - wait() Suspends the current thread if necessary until the condition
            has been turned on. If the condition is already on, the function
            simply exits. After wait() the condition is switched off.
     - signal() Indicates that the condition is met, turns the condition on,
            waking up the waiting thread. Creates no delay.
     - broadcast() Wakes up all the threads that are waiting on the condition.

    \sa cond_base

    \ingroup libutil_threads
 **/
class cond;


} // namespace libutil


#if defined(USE_PTHREADS)

#include "posix/cond_posix.h"
namespace libutil {
class cond : public cond_base<cond_posix> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS)

#include "windows/cond_windows.h"
namespace libutil {
class cond : public cond_base<cond_windows> { };
} // namespace libutil

#endif


#endif // LIBUTIL_COND_H
