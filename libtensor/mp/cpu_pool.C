#include <sstream>
#include "cpu_pool.h"
#include "mp_exception.h"

namespace libtensor {


const char *cpu_pool::k_clazz = "cpu_pool";


cpu_pool::cpu_pool(size_t ncpus) {

    m_cpus.resize(ncpus, 0);
    m_free.reserve(ncpus);
    for(size_t i = 0; i < ncpus; i++) {
        std::ostringstream ss;
        ss << i + 1;
        m_free.push_back(i);
        m_name.push_back(ss.str());
    }
}


size_t cpu_pool::acquire_cpu() {

    while(true) {
        {
            auto_lock<mutex> lock(m_lock);
            if(!m_free.empty()) {
                size_t cpuid = m_free.back();
                m_free.pop_back();
                m_cpus[cpuid] = 1;
                cpu_pool::start_timer(m_name[cpuid]);
                return cpuid;
            }
        }
        m_cond.wait();
    }
}


void cpu_pool::release_cpu(size_t cpuid) {

    static const char *method = "release_cpu(size_t)";

    {
        auto_lock<mutex> lock(m_lock);
        if(m_cpus[cpuid] == 0) {
            throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "bad_cpuid");
        }
        cpu_pool::stop_timer(m_name[cpuid]);
        m_cpus[cpuid] = 0;
        m_free.push_back(cpuid);
    }
    m_cond.signal();
}


} // namespace libtensor
