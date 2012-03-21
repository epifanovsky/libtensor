#ifndef LIBTENSOR_TASK_DISPATCHER_H
#define LIBTENSOR_TASK_DISPATCHER_H

#include <list>
#include <libutil/singleton.h>
#include "cpu_pool.h"
#include "task_queue.h"
#include "threads.h"

namespace libtensor {


/** \brief Contains the current task queue for the thread

    \ingroup libtensor_mp
 **/
struct current_task_queue {
    task_queue *tq;
    current_task_queue() : tq(0) { }
};


/** \brief Parallel task dispatcher

    \ingroup libtensor_mp
 **/
class task_dispatcher : public libutil::singleton<task_dispatcher> {
    friend class libutil::singleton<task_dispatcher>;

public:
    static const char *k_clazz; //!< Class name

public:
    typedef std::list<task_queue*>::iterator queue_id_t; //!< Queue ID type

private:
    mutex m_lock; //!< Lock
    cond m_alarm; //!< Alarm for the processing pool
    bool m_mp; //!< Use multiple threads
    task_queue m_root; //!< Root task queue
    std::list<task_queue*> m_tqs; //!< List of task queues
    volatile size_t m_ntasks; //!< Number of scheduled tasks
    volatile size_t m_nwaiting; //!< Number of threads waiting on alarm

protected:
    /** \brief Protected singleton constructor
     **/
    task_dispatcher();

public:
    //!	\name Interface to task batches
    //@{

    /**	\brief Creates a queue on the top of the stack and returns its
            identifier
     **/
    queue_id_t create_queue();

    /**	\brief Destroys the queue specified by its identifier
     **/
    void destroy_queue(queue_id_t &qid);

    /**	\brief Schedules a task in a queue. The queue must be valid and
            no threads should be waiting on it.
     **/
    void push_task(const queue_id_t &qid, task_i &task);

    /** \brief Waits until all tasks in the queue are over. Once there
            is a waiter on a queue, tasks cannot be scheduled there
     **/
    void wait_on_queue(const queue_id_t &qid);

    /**	\brief Waits until all tasks in the queue are over. Once there
            is a waiter on a queue, tasks cannot be scheduled there
     **/
    void wait_on_queue(const queue_id_t &qid, cpu_pool &cpus);

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
        \param cpus Pool of CPUs.
     **/
    void invoke_next(cpu_pool &cpus);

    //@}

private:
    bool invoke_next(task_queue &tq, cpu_pool &cpus);

};


} // namespace libtensor

#endif // LIBTENSOR_TASK_DISPATCHER_H
