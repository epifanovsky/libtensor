#include "task_dispatcher.h"
#include "worker.h"

namespace libtensor {


worker::worker(libvmm::cond &started, libvmm::mutex &cpu_lock) :

	m_started(started), m_cpu_lock(cpu_lock), m_term(false) {

}


worker::~worker() {

}


void worker::run() {

	m_started.signal();
	while(!m_term) {
		task_dispatcher::get_instance().wait_next();
		m_cpu_lock.lock();
		task_dispatcher::get_instance().invoke_next();
		m_cpu_lock.unlock();
	}
}


void worker::terminate() {

	m_term = true;
}


} // namespace libtensor
