#ifndef LIBTENSOR_TASK_QUEUE_H
#define LIBTENSOR_TASK_QUEUE_H

#include <deque>
#include <set>
#include <utility>
#include "threads.h"
#include "task_i.h"
#include "../exception.h"

namespace libtensor {


/** \brief Recursive task queue for multiple producers and multiple consumers

    Stores a queue of tasks pending execution and a list of all children
    queues.

    The producers insert tasks using push() and wait for the completion by
    calling wait().

    The consumers invoke is_empty() to see if the queue has more tasks and
    extract the next task using pop(). After the task has been extracted, its
    status is marked as "in progress" until the queue is notified of its
    completion via finished().

    \ingroup libtensor_mp
 **/
class task_queue {
public:
    static const char *k_clazz; //!< Class name

private:
    std::deque<task_i*> m_q; //!< Queue
    size_t m_queued; //!< Number of queued tasks (this + children)
    size_t m_inprogress; //!< Number of tasks in progress (this + children)
    task_queue *m_parent; //!< Parent queue
    std::set<task_queue*> m_children; //!< Children queues
    exception *m_exc; //!< Exception
    mutable mutex m_lock; //!< Mutex lock
    mutable cond m_sig; //!< Completion signal

public:
    //! \name Construction and destruction
    //@{

    /** \brief Initializes the task queue
        \param parent Parent queue (default 0 for the root queue).
     **/
    task_queue(task_queue *parent = 0);

    /** \brief Destroys the task queue
     **/
    ~task_queue();

    //@}


    //! \name Interface for the producers
    //@{

    /** \brief Returns the parent queue or null pointer
     **/
    task_queue *get_parent() {
        return m_parent;
    }

    /** \brief Inserts a task at the back of the queue
        \param task Task.
     **/
    void push(task_i *task);

    /** \brief Waits for the completion of all the tasks in this queue and all
            the children
     **/
    void wait();

    //@}


    //! \name Interface for the consumers
    //@{

    /** \brief Returns true if this and all the children queues do not have any
            more queued tasks
     **/
    bool is_empty() const;

    /** \brief Extracts and returns the task at the back of the queue
     **/
    std::pair<task_queue*, task_i*> pop();

    /** \brief Marks a task as finished
     **/
    void finished(task_i &task);

    /** \brief Sets the exception associated with the queue
     **/
    void set_exception(exception &exc);

    //@}

private:
    /** \brief Adds a task queue to the list of the children
     **/
    void add_child(task_queue *child);

    /** \brief Removes a task queue from the list of the children
     **/
    void remove_child(task_queue *child);

};


} // namespace libtensor

#endif // LIBTENSOR_TASK_QUEUE_H
