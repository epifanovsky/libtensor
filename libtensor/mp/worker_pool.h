#ifndef LIBTENSOR_WORKER_POOL_H
#define LIBTENSOR_WORKER_POOL_H

#include <libutil/singleton.h>
#include "cpu_pool.h"
#include "worker_group.h"

namespace libtensor {


/** \brief Maintains a pool of worker threads

    In a pool of workers, the worker threads compete for limited CPU resources.
    Tasks executed by the workers are able to switch between idling and
    CPU-intensive cycles thus allowing overlap.

    \ingroup libtensor_mp
 **/
class worker_pool : public libutil::singleton<worker_pool> {
    friend class libutil::singleton<worker_pool>;

public:
    static const char *k_clazz; //!< Class name

private:
    bool m_init; //!< The pool is active (true) or inactive (false)
    cpu_pool *m_cpus; //!< Pool of CPUs
    worker_group *m_wg; //!< Group of worker threads

protected:
    /** \brief Protected singleton constructor
     **/
    worker_pool();

public:
    /** \brief Shuts down and destroys the pool
     **/
    virtual ~worker_pool();

public:
    /** \brief Initializes the pool with the given numbers of CPUs and threads
        \param ncpus Number of CPUs.
        \param nthreads Number of threads.
     **/
    void init(size_t ncpus, size_t nthreads);

    /** \brief Shuts down the pool and terminates all the threads
     **/
    void shutdown();

    /** \brief Returns whether the worker pool is initialized and running
     **/
    bool is_running() const {
        return m_init;
    }

    /** \brief Returns the pool of CPUs
     **/
    cpu_pool &get_cpus();

};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_POOL_H
