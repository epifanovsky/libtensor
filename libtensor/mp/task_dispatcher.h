#ifndef LIBTENSOR_TASK_DISPATCHER_H
#define LIBTENSOR_TASK_DISPATCHER_H

#include <list>
#include <libvmm/singleton.h>
#include "threads.h"
#include "task_queue.h"

namespace libtensor {


/**	\brief Parallel task dispatcher

	\ingroup libtensor_mp
 **/
class task_dispatcher : public libvmm::singleton<task_dispatcher> {
	friend class libvmm::singleton<task_dispatcher>;

public:
	static const char *k_clazz; //!< Class name

private:
	struct queue {
		task_queue q; //!< Queue
		size_t nrunning; //!< Number of running tasks
		bool finalized; //!< Finalized queue
		bool destroyed; //!< Destroyed queue
		exception *exc; //!< Exception
		queue() : nrunning(0), finalized(false), destroyed(false),
			exc(0) { }
	};

public:
	typedef std::list<queue*>::iterator queue_id_t; //!< Queue ID type

private:
	spinlock m_lock; //!< Lock
	cond m_alarm; //!< Alarm for the processing pool
	std::list<queue*> m_stack; //!< Stack of queues
	volatile size_t m_ntasks; //!< Number of scheduled tasks
	volatile size_t m_nwaiting; //!< Number of threads waiting on alarm

protected:
	task_dispatcher() : m_ntasks(0), m_nwaiting(0) { }

public:
	//!	\name Interface to task batches
	//@{

	/**	\brief Creates a queue on the top of the stack and returns its
			identificator
	 **/
	queue_id_t create_queue();

	/**	\brief Destroys the queue specified by its identificator
	 **/
	void destroy_queue(queue_id_t &qid);

	/**	\brief Schedules a task in a queue. The queue must be valid and
			no threads should be waiting on it.
	 **/
	void push_task(queue_id_t &qid, task_i &task);

	/**	\brief Waits until all tasks in the queue are over. Once there
			is a waiter on a queue, tasks cannot be scheduled there
	 **/
	void wait_on_queue(queue_id_t &qid);

	//@}


	//!	\name Interface to the worker pool
	//@{

	/**	\brief Wakes up all the threads waiting on the alarm
	 **/
	void set_off_alarm();

	/**	\brief Waits until there is at least one task scheduled
	 **/
	void wait_next();

	/**	\brief Executes the next task in the queue or simply returns
	 **/
	void invoke_next();

	//@}

private:
	void erase_queue_from_list(queue_id_t &qid);

};


} // namespace libtensor

#endif // LIBTENSOR_TASK_DISPATCHER_H
