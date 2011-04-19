#include "task_dispatcher.h"
#include "worker.h"

namespace libtensor {


const char *worker::k_clazz = "worker";


worker::worker(libvmm::cond &started, libvmm::mutex &cpu_lock) :

	m_started(started), m_cpu_lock(cpu_lock), m_term(false) {

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
		m_cpu_lock.lock();
		start_timer("work");
		task_dispatcher::get_instance().invoke_next();
		stop_timer("work");
		m_cpu_lock.unlock();
	}
	stop_timer();
}


void worker::terminate() {

	m_term = true;
}


} // namespace libtensor
