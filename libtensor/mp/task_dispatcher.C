#include "../exception.h"
#include "mp_exception.h"
#include "task_dispatcher.h"

namespace libtensor {


const char *task_dispatcher::k_clazz = "task_dispatcher";


task_dispatcher::queue_id_t task_dispatcher::create_queue() {

	libvmm::auto_spinlock lock(m_lock);

	return m_stack.insert(m_stack.end(), new queue);
}


void task_dispatcher::destroy_queue(queue_id_t &qid) {

	static const char *method = "destroy_queue(queue_id_t&)";

	libvmm::auto_spinlock lock(m_lock);

	if(qid == m_stack.end()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"qid");
	}

	queue *q = *qid;
	erase_queue_from_list(qid);
	delete q;
}


void task_dispatcher::push_task(queue_id_t &qid, task_i &task) {

	static const char *method = "push_task(queue_id_t&, task_i&)";

	libvmm::auto_spinlock lock(m_lock);

	if(qid == m_stack.end()) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"qid");
	}

	queue &q = **qid;
	if(q.finalized) {
		throw mp_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
			"bad_queue_state: finalized");
	}

	q.q.push(task);
	m_ntasks++;
	m_alarm.signal();
}


void task_dispatcher::wait_on_queue(queue_id_t &qid) {

	static const char *method = "wait_on_queue(queue_id_t&)";

	{
		libvmm::auto_spinlock lock(m_lock);

		queue &q = **qid;
		if(q.finalized) {
			throw mp_exception(g_ns, k_clazz, method, __FILE__,
				__LINE__, "bad_queue_state: finalized");
		}
		q.finalized = true;
	}

	while(true) {
		bool done = false;
		{
			libvmm::auto_spinlock lock(m_lock);
			queue &q = **qid;
			done = q.q.is_empty() && q.nrunning == 0;
		}
		if(done) break;
		invoke_next();
	}

	{
		libvmm::auto_spinlock lock(m_lock);

		queue &q = **qid;
		if(q.exc != 0) {
			try {
				q.exc->rethrow();
			} catch(exception &e) {
				delete q.exc; q.exc = 0;
				throw;
			}
		}
	}
}


void task_dispatcher::set_off_alarm() {

	while(m_nwaiting > 0) {
		m_alarm.signal();
	}
}


void task_dispatcher::wait_next() {

	{
		libvmm::auto_spinlock lock(m_lock);
		if(m_ntasks != 0) return;
		m_nwaiting++;
	}
	m_alarm.wait();
	{
		libvmm::auto_spinlock lock(m_lock);
		m_nwaiting--;
	}
}


void task_dispatcher::invoke_next() {

	queue *q = 0;
	task_i *task = 0;

	{
		libvmm::auto_spinlock lock(m_lock);

		std::list<queue*>::reverse_iterator i = m_stack.rbegin();
		for(; i != m_stack.rend() && (*i)->q.is_empty(); i++);
		if(i == m_stack.rend()) return;

		q = *i;
		task = &q->q.pop();
		q->nrunning++;
		m_ntasks--;
	}

	exception *exc = 0;
	try {
		task->perform();
	} catch(exception &e) {
		exc = e.clone();
	} catch(...) {
	}

	{
		libvmm::auto_spinlock lock(m_lock);
		q->nrunning--;
		if(exc) {
			if(q->exc == 0) q->exc = exc;
			else delete exc;
		}
	}
}


void task_dispatcher::erase_queue_from_list(queue_id_t &qid) {

	m_stack.erase(qid);
	qid = m_stack.end();
}


} // namespace libtensor
