#ifndef LIBUTIL_WORKER_H
#define LIBUTIL_WORKER_H

#include <libutil/threads/cond.h>
#include <libutil/threads/mutex.h>
#include <libutil/threads/thread.h>

namespace libutil {


class thread_pool;


/** \brief Worker thread

    \ingroup libutil_thread_pool
 **/
class worker : public thread {
private:
    thread_pool &m_pool; //!< Thread pool
    cond *m_start_cond; //!< Start conditional

public:
    /** \brief Initializes the worker thread
        \param pool Thread pool to which this worker belongs.
        \param c Thread start conditional.
     **/
    worker(thread_pool &pool, cond *c) : m_pool(pool), m_start_cond(c)
    { }

    /** \brief Runs the worker thread
     **/
    virtual void run();

    /** \brief Notifies the thread pool that thw worker is ready
     **/
    void notify_ready();

};


} // namespace libutil

#endif // LIBUTIL_WORKER_H

