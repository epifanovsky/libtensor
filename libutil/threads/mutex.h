#ifndef LIBUTIL_MUTEX_H
#define LIBUTIL_MUTEX_H

#include "base/mutex_base.h"
#include "posix/mutex_posix.h"

namespace libutil {


/** \brief Mutex

    Mutexes prevents simultaneous access of two threads to a protected object.

    \ingroup libutil_threads
 **/
class mutex : public mutex_base<mutex_posix> { };


} // namespace libutil


#endif // LIBUTIL_MUTEX_H
