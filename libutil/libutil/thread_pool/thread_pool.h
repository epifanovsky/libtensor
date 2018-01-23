#ifndef LIBUTIL_THREAD_POOL_H
#define LIBUTIL_THREAD_POOL_H

#include <deque>
#include <map>
#include <vector>
#include <libutil/threads/mutex.h>
#include <libutil/threads/spinlock.h>
#include "task_iterator_i.h"
#include "task_observer_i.h"
#include "task_source.h"
#include "task_thief.h"
#include "worker.h"

namespace libutil {


/** \brief Thread pool

    \ingroup libutil_thread_pool
 **/
class thread_pool {
private:
    enum {
        WORKER_STATE_IDLE, WORKER_STATE_RUNNING
    };

    struct worker_info {
        int state;
        cond sig;
        cond cpu;
    };

    struct ts_stats {
        size_t ntasks;
        unsigned long totcost;
        ts_stats() : ntasks(0), totcost(0) { }
    };

private:
    size_t m_nthreads; //!< Max number of non-idle threads
    size_t m_ncpus; //!< Max number of running threads
    size_t m_nrunning; //!< Number of running threads
    size_t m_nwaiting; //!< Number of waiting threads
    std::map<worker*, worker_info*> m_winfo; //!< Info about each worker
    std::vector<worker*> m_all; //!< List of all workers
    std::vector<worker*> m_idle; //!< List of idle workers
    std::vector<worker*> m_running; //!< List of running threads
    std::vector<worker*> m_waiting; //!< List of waiting threads
    std::vector<worker*> m_waitingcpu; //!< List of threads waiting for CPU
    task_source *m_tsroot; //!< Root task source
    std::map<task_source*, ts_stats> m_tsstat; //!< Task source stats
    task_thief m_thief; //!< Task thief
    volatile bool m_term; //!< Termination flag
    spinlock m_mtx; //!< Mutex

public:
    /** \brief Creates a thread pool
        \param nthreads Limit on non-idle (running + waiting) threads.
        \param ncpus Limit on number of CPUs (running threads).
     **/
    thread_pool(size_t nthreads, size_t ncpus);

    /** \brief Destroys the thread pool
     **/
    ~thread_pool();

    /** \brief Terminates all threads
     **/
    void terminate();

    /** \brief Associates thread pool with the current thread. Only one thread
            pool can be associated with each thread
     **/
    void associate(worker *w = 0);

    /** \brief Dissociates thread pool from the current thread. Must be
            previously connected using associate()
     **/
    void dissociate();

    /** \brief Worker's main function (task loop)
        \param w Worker.
     **/
    void worker_main(worker *w);

    /** \brief Submits a collection of tasks to the thread pool currently
            associated with the current thread. Returns when all the tasks
            have been completed
     **/
    static void submit(task_iterator_i &ti, task_observer_i &to);

    /** \brief Allocates a CPU for the current thread (and waits for one
            to become available if necessary)
     **/
    static void acquire_cpu();

    /** \brief Deallocates current thread's CPU making it available for other
            threads
     **/
    static void release_cpu();

private:
    static void run_serial(task_iterator_i &ti, task_observer_i &to);

    void do_submit(task_iterator_i &ti, task_observer_i &to);
    void do_acquire_cpu(bool intask);
    void do_release_cpu(bool intask);

    void enqueue_local(std::deque<task_info> &lq, size_t maxn, spinlock &lqmtx);

    void create_idle_thread();
    void activate_idle_thread();
    void activate_waiting_thread();

    void add_to_list(worker *w, std::vector<worker*> &l);
    worker *pop_from_list(std::vector<worker*> &l);
    void remove_from_list(worker *w, std::vector<worker*> &l);

};


} // namespace libutil

#endif // LIBUTIL_THREAD_POOL_H
