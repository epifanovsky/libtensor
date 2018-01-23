#ifndef LIBUTIL_THREAD_H
#define LIBUTIL_THREAD_H

#include "base/thread_base.h"

namespace libutil {


/** \brief Thread

    \ingroup libutil_threads
 **/
class thread;


} // namespace libutil


#if defined(USE_PTHREADS)

#include "posix/thread_posix.h"
namespace libutil {
class thread : public thread_base<thread_posix> { };
} // namespace libutil


#elif defined(USE_WIN32_THREADS)

#include "windows/thread_windows.h"
namespace libutil {
class thread : public thread_base<thread_windows> { };
} // namespace libutil

#endif


#endif // LIBUTIL_THREAD_H
