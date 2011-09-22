#include "cpu_pool.h"

namespace libtensor {


cpu_pool::cpu_pool(size_t ncpus) {

    m_cpus.resize(ncpus, 0);
    m_free.reserve(ncpus);
    for(size_t i = 0; i < ncpus; i++) m_free.push_back(i);
}


size_t cpu_pool::acquire_cpu() {

    while(true) {
        {
            auto_spinlock lock(m_lock);
            if(!m_free.empty()) {
                size_t cpuid = m_free.back();
                m_free.pop_back();
                m_cpus[cpuid] = 1;
                return cpuid;
            }
        }
        m_cond.wait();
    }
}


void cpu_pool::release_cpu(size_t cpuid) {

    {
        auto_spinlock lock(m_lock);
        if(m_cpus[cpuid] == 0) {
            throw 0;
        }
        m_cpus[cpuid] = 0;
        m_free.push_back(cpuid);
    }
    m_cond.signal();
}


} // namespace libtensor
