#include "../exception.h"
#include "mp_exception.h"
#include "task_dispatcher.h"
#include "worker_pool.h"

namespace libtensor {


const char *worker_pool::k_clazz = "worker_pool";


void worker_pool::init(unsigned ngroups, unsigned nthreads) {

	static const char *method = "init(unsigned, ngroups)";

	if(nthreads == 0) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"nthreads");
	}
	if(m_init) {
		throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
			"running");
	}

	std::vector<cond*> sig(ngroups, 0);
	for(unsigned i = 0; i < ngroups; i++) {
		sig[i] = new cond;
		m_groups.push_back(new worker_group(nthreads, *sig[i]));
	}
	for(unsigned i = 0; i < ngroups; i++) {
		m_groups[i]->start();
	}
	for(unsigned i = 0; i < ngroups; i++) {
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

	unsigned ngroups = m_groups.size();
	for(unsigned i = 0; i < ngroups; i++) {
		m_groups[i]->terminate();
	}
	task_dispatcher::get_instance().set_off_alarm();
	for(unsigned i = 0; i < ngroups; i++) {
		m_groups[i]->join();
	}
	for(unsigned i = 0; i < ngroups; i++) {
		delete m_groups[i];
	}
	m_groups.clear();

	m_init = false;
}


} // namespace libtensor
