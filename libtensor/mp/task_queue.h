#ifndef LIBTENSOR_TASK_QUEUE_H
#define LIBTENSOR_TASK_QUEUE_H

#include <deque>
#include "task_i.h"

namespace libtensor {


/**	\brief Thread-safe task queue

	\ingroup libtensor_mp
 **/
class task_queue {
private:
	std::deque<task_i*> m_q; //!< Queue
	mutable libvmm::mutex m_lock; //!< Mutex lock

public:
	bool is_empty() const {
		libvmm::auto_lock lock(m_lock);
		return m_q.empty();
	}

	void push(task_i &task) {
		libvmm::auto_lock lock(m_lock);
		m_q.push_back(&task);
	}

	task_i &pop() {
		libvmm::auto_lock lock(m_lock);
		task_i *t = m_q.front();
		m_q.pop_front();
		return *t;
	}
};


} // namespace libtensor

#endif // LIBTENSOR_TASK_QUEUE_H
