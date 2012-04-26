#include "../exception.h"
#include "mp_exception.h"
#include "task_dispatcher.h"
#include "worker_pool.h"

namespace libtensor {


const char *worker_pool::k_clazz = "worker_pool";


worker_pool::worker_pool() : m_init(false), m_cpus(0), m_wg(0) {

}


worker_pool::~worker_pool() {

    if(m_init) shutdown();
}


void worker_pool::init(size_t ncpus, size_t nthreads) {

    static const char *method = "init(size_t, size_t)";

    if(ncpus == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "ncpus");
    }
    if(nthreads == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "nthreads");
    }
    if(m_init) {
        throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "running");
    }

    m_cpus = new cpu_pool(ncpus);

    m_wg = new worker_group(nthreads, *m_cpus);
    m_wg->start();

    m_init = true;
}


void worker_pool::shutdown() {

    static const char *method = "shutdown()";

    if(!m_init) {
        throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "not_running");
    }

    m_init = false;

    m_wg->terminate();
    task_dispatcher::get_instance().set_off_alarm();
    m_wg->join();
    delete m_wg;
    m_wg = 0;
    delete m_cpus;
    m_cpus = 0;
}


cpu_pool &worker_pool::get_cpus() {

    static const char *method = "get_cpus()";

    if(!m_init) {
        throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "not_running");
    }

    return *m_cpus;
}


} // namespace libtensor
