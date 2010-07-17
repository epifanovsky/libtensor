#include "../exception.h"
#include "mp_exception.h"
#include "task_dispatcher.h"
#include "worker_pool.h"

namespace libtensor {


const char *worker_pool::k_clazz = "worker_pool";


void worker_pool::init(unsigned nthreads) {

	static const char *method = "init(unsigned)";

	if(nthreads == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"nthreads");
	}
	if(m_init) {
		throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
			"running");
	}

	std::vector<libvmm::cond*> sig(nthreads, 0);
	for(unsigned i = 0; i < nthreads; i++) {
		sig[i] = new libvmm::cond;
		m_workers.push_back(new worker(*sig[i]));
	}
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->start();
	}
	for(unsigned i = 0; i < nthreads; i++) {
		sig[i]->wait();
	}

	m_init = true;
}


void worker_pool::shutdown() {

	static const char *method = "shutdown()";

	if(!m_init) {
		throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
			"not_running");
	}

	unsigned nthreads = m_workers.size();
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->terminate();
	}
	task_dispatcher::get_instance().set_off_alarm();
	for(unsigned i = 0; i < nthreads; i++) {
		m_workers[i]->join();
	}
	for(unsigned i = 0; i < nthreads; i++) {
		delete m_workers[i];
	}
	m_workers.clear();

	m_init = false;
}


} // namespace libtensor
