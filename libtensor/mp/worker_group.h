#ifndef LIBTENSOR_WORKER_GROUP_H
#define LIBTENSOR_WORKER_GROUP_H

#include <vector>
#include "worker.h"

namespace libtensor {


/** \brief Group of worker threads sharing a pool of CPUs

    \ingroup libtensor_mp
 **/
class worker_group {
private:
    cpu_pool &m_cpus; //!< Pool of CPUs
    std::vector<worker*> m_workers; //!< Worker threads
    std::vector<cond*> m_started; //!< Worker start signals

public:
    /** \brief Initializes the group of workers
        \param nthreads Number of threads in the group.
        \param cpus Pool of CPUs.
     **/
    worker_group(size_t nthreads, cpu_pool &cpus);

    /** \brief Starts all the threads in the group
     **/
    void start();

    /** \brief Sends a termination signal to all the threads in the group
     **/
    void terminate();

    /** \brief Waits until all the threads in the group have terminated
     **/
    void join();

};


} // namespace libtensor

#endif // LIBTENSOR_WORKER_GROUP_H
