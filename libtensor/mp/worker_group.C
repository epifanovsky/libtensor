#include "task_dispatcher.h"
#include "worker_group.h"
#include "worker.h"

namespace libtensor {


worker_group::worker_group(unsigned nthreads, libvmm::cond &started) {

	for(unsigned i = 0; i < nthreads; i++) {
		m_started.push_back(new libvmm::cond);
		m_workers.push_back(new worker(*m_started[i], m_cpu_lock));
	}

	started.signal();
}


void worker_group::start() {

	unsigned nthreads = m_workers.size();
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->start();
	}
	for(unsigned i = 0; i < nthreads; i++) {
		m_started[i]->wait();
	}
	for(unsigned i = 0; i < nthreads; i++) {
		delete m_started[i];
	}
	m_started.clear();
}


void worker_group::terminate() {

	unsigned nthreads = m_workers.size();
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->terminate();
	}
}


void worker_group::join() {

	unsigned nthreads = m_workers.size();
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->join();
	}
	for(unsigned i = 0; i < nthreads; i++) {
		delete m_workers[i];
	}
	m_workers.clear();
}


} // namespace libtensor
