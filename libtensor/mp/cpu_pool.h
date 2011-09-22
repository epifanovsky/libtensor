#ifndef LIBTENSOR_CPU_POOL_H
#define LIBTENSOR_CPU_POOL_H

#include <vector>
#include "threads.h"

namespace libtensor {


/** \brief Manages a pool of CPUs

    The CPU pool maintains a list of available processors and allocates them
    for specific computational tasks.

    acquire_cpu() is a locking method that allocates one CPU from the pool. If
    there are no free CPUs at the time of the call, it waits until a CPU becomes
    available. The CPU is acquired for a specific computation by the calling
    thread, which makes it unavailable for other threads until it is released
    back to the pool.

    release_cpu() returns a previously acquired CPU back to the pool.

    The CPU pool controls the flow such that the number of active threads
    doesn't exceed the total number of available CPUs.

    \ingroup libtensor_mp
 **/
class cpu_pool {
private:
    std::vector<unsigned> m_cpus; //!< List of all CPUs marked as free/busy
    std::vector<size_t> m_free; //!< List of free CPUs
    spinlock m_lock; //!< Lock for the lists
    cond m_cond; //!< Conditional to wait for CPUs

public:
    /** \brief Initializes the pool of CPUs
        \param ncpus Number of CPUs.
     **/
    cpu_pool(size_t ncpus);

    /** \brief Acquires a CPU from the pool (locking)
        \return CPU identifier.
     **/
    size_t acquire_cpu();

    /** \brief Releases a CPU back to the pool
        \param cpuid CPU identifier obtained from acquire_cpu().
     **/
    void release_cpu(size_t cpuid);

};


} // namespace libtensor

#endif // LIBTENSOR_CPU_POOL_H
