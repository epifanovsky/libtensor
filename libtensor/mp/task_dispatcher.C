#include "task_dispatcher.h"

namespace libtensor {


task_dispatcher::queue_id_t task_dispatcher::create_queue() {

	libvmm::auto_lock lock(m_lock);

	return m_stack.insert(m_stack.end(), new queue);
}


void task_dispatcher::destroy_queue(queue_id_t &qid) {

	libvmm::auto_lock lock(m_lock);

	if(qid == m_stack.end()) return;

	queue *q = *qid;
	if(q->running) {
		q->destroyed = true;
	} else {
		erase_queue_from_list(qid);
		delete q;
	}
}


void task_dispatcher::push_task(queue_id_t &qid, task_i &task) {

	libvmm::auto_lock lock(m_lock);

	if(q == m_stack.end()) return;

	queue &q = **qid;
	if(q.waiting || q.destroyed) {
		// throw bad_queue exception here
	}

	q.q.push(task);
	if(m_ntasks++ == 0) m_alarm.signal();
}


void task_dispatcher::wait_on_queue(queue_id_t &qid) {

	libvmm::cond *sig = 0;

	{
		libvmm::auto_lock lock(m_lock);

		queue &q = **qid;
		if(q.waiting || q.destroyed) {
			// throw bad_queue exception here
		}

		q.waiting = true;
		sig = &q.sig;
	}
	sig->wait();
}


void task_dispatcher::wait_next() {

	bool need_wait = false;
	{
		libvmm::auto_lock lock(m_lock);
		need_wait = m_ntasks == 0;
	}
	if(need_wait) m_alarm.wait();
}


void task_dispatcher::invoke_next() {

	libvmm::auto_lock lock(m_lock);

	std::list<queue*>::reverse_iterator i = m_stack.rbegin();
	for(; i != m_stack.rend() && (*i)->q.is_empty(); i++);
	if(i == m_stack.rend()) return;

	queue &q = **i;
	task_i &task = q.q.pop();
	m_ntasks--;
	lock.unlock();
	try {
		task.perform();
	} catch(...) {
		lock.lock();
		throw;
	}
	lock.lock();
}


void task_dispatcher::erase_queue_from_list(queue_id_t &qid) {

	m_stack.erase(qid);
	qid = m_stack.end();
}


} // namespace libtensor
