#include "task_dispatcher.h"
#include "worker.h"

namespace libtensor {


worker::worker(libvmm::cond &started) : m_started(started), m_term(false) {

}


worker::~worker() {

}


void worker::run() {

	m_started.signal();
	while(!m_term) {
		task_dispatcher::get_instance().wait_next();
		task_dispatcher::get_instance().invoke_next();
	}
}


void worker::terminate() {

	m_term = true;
}


} // namespace libtensor
