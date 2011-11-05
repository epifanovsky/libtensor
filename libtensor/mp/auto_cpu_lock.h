#ifndef LIBTENSOR_AUTO_CPU_LOCK_H
#define LIBTENSOR_AUTO_CPU_LOCK_H

#include "cpu_pool.h"

namespace libtensor {


/** \brief Automatic scope CPU lock

    This class acquires a CPU from cpu_pool upon creation and releases the CPU
    upon destruction.

    \sa cpu_pool

    \ingroup libtensor_mp
 **/
class auto_cpu_lock {
private:
    cpu_pool &m_cpus; //!< CPU pool
    size_t m_cpuid; //!< CPU id

public:
    /** \brief Acquires a CPU from the given pool
     **/
    auto_cpu_lock(cpu_pool &cpus) :

        m_cpus(cpus) {

        m_cpuid = m_cpus.acquire_cpu();
    }

    /** \brief Releases the CPU back into the pool
     **/
    ~auto_cpu_lock() throw() {
        try {
            m_cpus.release_cpu(m_cpuid);
        } catch(...) {

        }
    }

};


} // namespace libtensor

#endif // LIBTENSOR_AUTO_CPU_LOCK_H
