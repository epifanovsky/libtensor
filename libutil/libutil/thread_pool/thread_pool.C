#include <algorithm>
#include <deque>
#include <memory>
#include <libutil/exceptions/util_exceptions.h>
#include <libutil/threads/auto_lock.h>
#include <libutil/threads/tls.h>
#include "thread_pool_info.h"
#include "thread_pool.h"
#include "unknown_exception.h"

namespace libutil {


thread_pool::thread_pool(size_t nthreads, size_t ncpus) :
    m_nthreads(nthreads), m_ncpus(ncpus), m_nrunning(0), m_nwaiting(0),
    m_tsroot(0), m_term(false) {

    for(size_t i = 0; i < nthreads; i++) create_idle_thread();
}


thread_pool::~thread_pool() {

    terminate();
    for(size_t i = 0; i < m_all.size(); i++) delete m_all[i];
}


void thread_pool::terminate() {

    {
        auto_lock<spinlock> lock(m_mtx);
        if(m_term) return;
        m_term = true;
    }

    worker *w;
    do {
        w = 0;
        {
            auto_lock<spinlock> lock(m_mtx);
            if(!m_winfo.empty()) {
                w = m_winfo.begin()->first;
                worker_info *wi = m_winfo.begin()->second;
                wi->sig.signal();
                wi->cpu.signal();
            }
        }
        if(w) w->join();
    } while(w);

    dissociate();
}


void thread_pool::associate(worker *w) {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();
    tpinfo.pool = this;
    tpinfo.tsrc = 0;
    tpinfo.w = w;
}


void thread_pool::dissociate() {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();
    tpinfo.pool = 0;
    tpinfo.tsrc = 0;
    tpinfo.w = 0;
}


void thread_pool::submit(task_iterator_i &ti, task_observer_i &to) {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();
    if(tpinfo.pool == 0) run_serial(ti, to);
    else tpinfo.pool->do_submit(ti, to);
}


void thread_pool::acquire_cpu() {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();
    if(tpinfo.pool == 0) return;
    tpinfo.pool->do_acquire_cpu(false);
}


void thread_pool::release_cpu() {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();
    if(tpinfo.pool == 0) return;
    tpinfo.pool->do_release_cpu(false);
}


void thread_pool::run_serial(task_iterator_i &ti, task_observer_i &to) {

    while(ti.has_more()) {
        task_i *tsk = ti.get_next();
        to.notify_start_task(tsk);
        tsk->perform();
        to.notify_finish_task(tsk);
    }
}


void thread_pool::do_submit(task_iterator_i &ti, task_observer_i &to) {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();

    task_source *ts_parent = tpinfo.tsrc;
    task_source ts(ts_parent, ti, to);
    {
        auto_lock<spinlock> lock(m_mtx);
        if(ts_parent == 0) m_tsroot = &ts;
    }
    tpinfo.tsrc = &ts;

    do_release_cpu(true);
    ts.wait();
    do_acquire_cpu(true);

    tpinfo.tsrc = ts_parent;
    {
        auto_lock<spinlock> lock(m_mtx);
        if(ts_parent == 0) m_tsroot = 0;
        m_tsstat.erase(&ts);
    }

    ts.rethrow_exceptions();
}


void thread_pool::do_acquire_cpu(bool intask) {

    {
        auto_lock<spinlock> lock(m_mtx);
        if(m_term) return;
    }

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();

    if(tpinfo.w == 0) return;

    bool done = false;
    cond *c = 0;
    {
        auto_lock<spinlock> lock(m_mtx);
        if(m_nrunning < m_ncpus) {
            remove_from_list(tpinfo.w, m_waiting);
            add_to_list(tpinfo.w, m_running);
            m_nrunning++;
            done = true;
        } else {
            remove_from_list(tpinfo.w, m_waiting);
            add_to_list(tpinfo.w, m_waitingcpu);
            c = &m_winfo[tpinfo.w]->cpu;
        }
    }
    while(!done && !m_term) {
        c->wait();
        {
            auto_lock<spinlock> lock(m_mtx);
            done = (m_winfo[tpinfo.w]->state == WORKER_STATE_RUNNING);
            if(done && !intask) m_nwaiting--;
        }
    }
}


void thread_pool::do_release_cpu(bool intask) {

    {
        auto_lock<spinlock> lock(m_mtx);
        if(m_term) return;
    }

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();

    if(tpinfo.w != 0) {
        bool create_idle = false;
        {
            auto_lock<spinlock> lock(m_mtx);
            create_idle = m_idle.empty();
//            create_idle = (m_idle.empty() &&
//                m_nrunning + m_nwaiting <= m_nthreads);
        }
        if(create_idle) create_idle_thread();
        {
            auto_lock<spinlock> lock(m_mtx);
            remove_from_list(tpinfo.w, m_running);
            add_to_list(tpinfo.w, m_waiting);
            m_nrunning--;
            if(!intask) m_nwaiting++;
        }
    }
    {
        auto_lock<spinlock> lock(m_mtx);
        if(m_nrunning < m_ncpus && !m_waitingcpu.empty()) {
            activate_waiting_thread();
            m_nrunning++;
        } else if(m_nrunning < m_ncpus && !m_idle.empty()) {
            activate_idle_thread();
            m_nrunning++;
        }
    }
}


void thread_pool::worker_main(worker *w) {

    thread_pool_info &tpinfo = tls<thread_pool_info>::get_instance().get();

    associate(w);

    worker_info winfo;
    winfo.state = WORKER_STATE_IDLE;
    std::map<worker*, worker_info*>::iterator iw;
    {
        auto_lock<spinlock> lock(m_mtx);
        iw = m_winfo.insert(std::make_pair(w, &winfo)).first;
        add_to_list(w, m_idle);
    }

    w->notify_ready();

    std::deque<task_info> lq; // Local queue
    spinlock lqmtx; //!< Lock on local queue (need for task stealing)
    const size_t lqlen = 4; // Number of tasks in local queue

    m_thief.register_queue(lq, lqmtx);

    bool good = true, first_task = true;

    while(good) {

        if(winfo.state == WORKER_STATE_IDLE) {

            winfo.sig.wait();
            first_task = true;

            {
                auto_lock<spinlock> lock(m_mtx);
                if(m_term) good = false;
            }

        } else if(winfo.state == WORKER_STATE_RUNNING) {

            if(first_task) {
                auto_lock<spinlock> lock(m_mtx);
                enqueue_local(lq, lqlen, lqmtx);
                first_task = false;
            }

            while(true) {

                task_info tinfo;
                {
                    auto_lock<spinlock> lockq(lqmtx);
                    if(lq.empty()) break;
                    tinfo = lq.front();
                    lq.pop_front();
                }

                //  Run next task
                tpinfo.tsrc = tinfo.tsrc;
                tinfo.tsrc->notify_start_task(tinfo.tsk);
                try {
                    tinfo.tsk->perform();
                } catch(rethrowable_i &e) {
                    tinfo.tsrc->notify_exception(tinfo.tsk, e);
                } catch(...) {
                    tinfo.tsrc->notify_exception(tinfo.tsk,
                        unknown_exception());
                }
                tinfo.tsrc->notify_finish_task(tinfo.tsk);
                tpinfo.tsrc = 0;
            }

            {
                auto_lock<spinlock> lock(m_mtx);

                //  Yield if another thread is waiting for CPU
                bool yield = !m_waitingcpu.empty();

                //  Pull next batch of tasks if still running
                if(!m_term && !yield && winfo.state == WORKER_STATE_RUNNING) {
                    enqueue_local(lq, lqlen, lqmtx);
                }

                bool empty_lq;
                {
                    auto_lock<spinlock> lqlock(lqmtx);
                    empty_lq = lq.empty();
                }

                //  Go idle if no more tasks in queue
                if(empty_lq) {
                    remove_from_list(w, m_running);
                    add_to_list(w, m_idle);
                    winfo.state = WORKER_STATE_IDLE;
                    m_nrunning--;
                }

                if(yield) {
                    activate_waiting_thread();
                    m_nrunning++;
                }

                if(m_term) good = false;
            }
        }
    }

    m_thief.unregister_queue(lq);

    {
        auto_lock<spinlock> lock(m_mtx);
        m_winfo.erase(iw);
    }

    dissociate();
}


void thread_pool::enqueue_local(std::deque<task_info> &lq, size_t maxn,
    spinlock &lqmtx) {

    //  Fills a queue with tasks based on their count and cost.
    //  If not enough stats from the task source have been gathered,
    //  at most maxn tasks will be enqueued.
    //  If enough stats are available from the task source, the total cost
    //  of enqueued tasks will be roughly equal to the average cost of
    //  maxn tasks from that source.
    //  The queue is assumed to be empty on entry.

    if(!m_tsroot) return;
    task_source *src = m_tsroot->get_current();

    size_t nadded = 0;

    if(src) {

        ts_stats &tss = m_tsstat[src];
        task_info tinfo;
        tinfo.tsrc = src;

        unsigned long avgcost =
            tss.ntasks > 2 * maxn ? tss.totcost / tss.ntasks : 0;
        unsigned long maxcost = avgcost * maxn;
        unsigned long cost = 0;

        while(cost > 0 && maxcost > 0 ? cost < maxcost : nadded < maxn) {

            task_i *t = src->extract_task();
            if(!t) break;
            unsigned long c = t->get_cost();
            cost += c;
            tss.ntasks++; tss.totcost += c;
            tinfo.tsk = t;
            {
                auto_lock<spinlock> lockq(lqmtx);
                lq.push_back(tinfo);
                nadded++;
            }
        }
    }

    //  If there are no more tasks left in the source, try stealing from
    //  another thread

    if(nadded == 0) {

        task_info tinfo;
        m_thief.steal_task(tinfo);
        if(tinfo.tsrc) {
            auto_lock<spinlock> lockq(lqmtx);
            lq.push_back(tinfo);
            nadded++;
        }
    }

    //  Use this as an opportunity to spawn more threads if appropriate

    if((m_tsroot->get_current() || nadded > 0) &&
            m_nrunning < m_ncpus && !m_idle.empty()) {
        activate_idle_thread();
        m_nrunning++;
    }
}


void thread_pool::create_idle_thread() {

    cond c;
    worker *w = new worker(*this, &c);
    {
        auto_lock<spinlock> lock(m_mtx);
        add_to_list(w, m_all);
    }
    w->start();
    c.wait();
}


void thread_pool::activate_idle_thread() {

    worker *w1 = pop_from_list(m_idle);
    worker_info *wi1 = m_winfo[w1];
    wi1->state = WORKER_STATE_RUNNING;
    add_to_list(w1, m_running);
    wi1->sig.signal();
}


void thread_pool::activate_waiting_thread() {

    worker *w1 = pop_from_list(m_waitingcpu);
    worker_info *wi1 = m_winfo[w1];
    wi1->state = WORKER_STATE_RUNNING;
    add_to_list(w1, m_running);
    wi1->cpu.signal();
}


void thread_pool::add_to_list(worker *w, std::vector<worker*> &l) {

    l.push_back(w);
}


worker *thread_pool::pop_from_list(std::vector<worker*> &l) {

    worker *w = l.back();
    l.pop_back();
    return w;
}


void thread_pool::remove_from_list(worker *w, std::vector<worker*> &l) {

    std::vector<worker*>::iterator i = std::find(l.begin(), l.end(), w);
    l.erase(i);
}


} // namespace libutil

