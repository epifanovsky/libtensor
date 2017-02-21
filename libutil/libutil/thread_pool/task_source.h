#ifndef LIBUTIL_TASK_SOURCE_H
#define LIBUTIL_TASK_SOURCE_H

#include <list>
#include <libutil/exceptions/rethrowable_i.h>
#include <libutil/threads/cond.h>
#include <libutil/threads/mutex.h>
#include "task_info.h"
#include "task_iterator_i.h"
#include "task_observer_i.h"

namespace libutil {


/** \brief Thread-safe task source

    The task source maintains an hierarchy of task queues and thread-safe
    methods to access the queues.

    Each task source in the hierarchy corresponds to a task iterator.

    Working with the task source, the user shall first obtain the current task
    source from the root of the hierarchy using get_current() and then request
    tasks from that source using extract_task().

    \ingroup libutil_thread_pool
 **/
class task_source {
private:
    task_source *m_parent; //!< Parent task source
    std::list<task_source*> m_children; //!< Children task sources
    const rethrowable_i *m_exc; //!< First exception
    task_iterator_i &m_ti; //!< Task iterator
    task_observer_i &m_to; //!< Task observer
    size_t m_npending; //!< Number of tasks about to be run
    size_t m_nrunning; //!< Number of currently running tasks
    mutex m_mtx; //!< Mutex
    cond m_alldone; //!< All done signal

public:
    /** \brief Initializes the task source
     **/
    task_source(task_source *parent, task_iterator_i &ti, task_observer_i &to);

    /** \brief Destroys the task source
     **/
    ~task_source();

    /** \brief Waits for all the tasks from this source and its children
            to complete
     **/
    void wait();

    /** \brief Rethrows exceptions occurred in the worker threads or does
            nothing if no exceptions are pending
     **/
    void rethrow_exceptions();

    /** \brief Returns the current task source or null if there are no sources
            with enqueued tasks
     **/
    task_source *get_current();

    /** \brief Returns the next task from the source
     **/
    task_i *extract_task();

    /** \brief Notifies the task source that a task has been started
     **/
    void notify_start_task(task_i *t);

    /** \brief Notifies the task source that a task has been finished
     **/
    void notify_finish_task(task_i *t);

    /** \brief Notifies the task source that an exception has been thrown while
            running a task
     **/
    void notify_exception(task_i *t, const rethrowable_i &e);

private:
    /** \brief Adds a child task source
     **/
    void add_child(task_source *ts);

    /** \brief Removes a child task source
     **/
    void remove_child(task_source *ts);

    /** \brief Checks if all tasks have completed (thread-safe)
     **/
    bool is_alldone();

    /** \brief Checks if all tasks have completed
     **/
    bool is_alldone_unsafe();

};


} // namespace libutil

#endif // LIBUTIL_TASK_SOURCE_H

