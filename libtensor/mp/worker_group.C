#include "task_dispatcher.h"
#include "worker_group.h"
#include "worker.h"

namespace libtensor {


worker_group::worker_group(size_t nthreads, cpu_pool &cpus) :

    m_cpus(cpus) {

    for(size_t i = 0; i < nthreads; i++) {
        m_started.push_back(new cond);
        m_workers.push_back(new worker(*m_started[i], m_cpus));
    }
}


void worker_group::start() {

    size_t nthreads = m_workers.size();
    for(size_t i = 0; i < nthreads; i++) {
        m_workers[i]->start();
    }
    for(size_t i = 0; i < nthreads; i++) {
        m_started[i]->wait();
    }
    for(size_t i = 0; i < nthreads; i++) {
        delete m_started[i];
    }
    m_started.clear();
}


void worker_group::terminate() {

    size_t nthreads = m_workers.size();
    for(size_t i = 0; i < nthreads; i++) {
        m_workers[i]->terminate();
    }
}


void worker_group::join() {

    size_t nthreads = m_workers.size();
    for(size_t i = 0; i < nthreads; i++) {
        m_workers[i]->join();
    }
    for(size_t i = 0; i < nthreads; i++) {
        delete m_workers[i];
    }
    m_workers.clear();
}


} // namespace libtensor
