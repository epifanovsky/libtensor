#include <libutil/threads/auto_lock.h>
#include "thread_pool.h"
#include "worker.h"

namespace libutil {


void worker::run() {

    m_pool.worker_main(this);
}


void worker::notify_ready() {

    if(m_start_cond) m_start_cond->signal(); // Notify the thread pool
    m_start_cond = 0;
}


} // namespace libutil

