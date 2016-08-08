#ifndef LIBUTIL_TASK_THIEF_H
#define LIBUTIL_TASK_THIEF_H

#include <deque>
#include <map>
#include <libutil/threads/mutex.h>
#include <libutil/threads/spinlock.h>
#include "task_info.h"

namespace libutil {


/** \brief Steals tasks from workers' local queues

    \ingroup libutil_thread_pool
 **/
class task_thief {
private:
    std::map< std::deque<task_info>*, spinlock* > m_queues; //!< Victims
    std::map< std::deque<task_info>*, spinlock* >::iterator m_i;
    spinlock m_mtx; //!< Lock

public:
    /** \brief Initializes the task thief
     **/
    task_thief();

    /** \brief Adds a candidate victim for theft
     **/
    void register_queue(std::deque<task_info> &lq, spinlock &lqmtx);

    /** \brief Removes a queue from the list of candidates
     **/
    void unregister_queue(std::deque<task_info> &lq);

    /** \brief Steals a task from one of the victims
     **/
    void steal_task(task_info &tinfo);

};


} // namespace libutil

#endif // LIBUTIL_TASK_THIEF_H

