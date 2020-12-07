#ifndef LIBUTIL_THREAD_H
#define LIBUTIL_THREAD_H

#include "base/thread_base.h"
#include "posix/thread_posix.h"

namespace libutil {


/** \brief Thread

    \ingroup libutil_threads
 **/
class thread : public thread_base<thread_posix> { };


} // namespace libutil

#endif // LIBUTIL_THREAD_H
