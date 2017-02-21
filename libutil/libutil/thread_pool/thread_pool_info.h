#ifndef LIBUTIL_THREAD_POOL_INFO_H
#define LIBUTIL_THREAD_POOL_INFO_H

#include "task_source.h"
#include "thread_pool.h"

namespace libutil {


/** \brief Contains worker-specific info about the thread pool

    \ingroup libutil_thread_pool
 **/
struct thread_pool_info {

    thread_pool *pool; //!< Worker owner
    task_source *tsrc; //!< Current source of tasks
    worker *w; //!< Worker

    thread_pool_info() : pool(0), tsrc(0), w(0) { }

};


} // namespace libutil

#endif // LIBUTIL_THREAD_POOL_INFO_H

