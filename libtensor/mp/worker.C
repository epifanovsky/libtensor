#include "task_dispatcher.h"
#include "worker.h"

namespace libtensor {


const char *worker::k_clazz = "worker";


worker::worker(cond &started, cpu_pool &cpus) :

    m_started(started), m_cpus(cpus), m_term(false) {

}


worker::~worker() {

}


void worker::run() {

    m_started.signal();

    start_timer();
    while(!m_term) {
        start_timer("wait");
        task_dispatcher::get_instance().wait_next();
        stop_timer("wait");
        start_timer("work");
        task_dispatcher::get_instance().invoke_next(m_cpus);
        stop_timer("work");
    }
    stop_timer();
}


void worker::terminate() {

    m_term = true;
}


} // namespace libtensor
