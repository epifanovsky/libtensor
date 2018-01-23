#include <libutil/threads/auto_lock.h>
#include "task_thief.h"

namespace libutil {


task_thief::task_thief() : m_i(m_queues.end()) {

}


void task_thief::register_queue(std::deque<task_info> &lq, spinlock &lqmtx) {

    auto_lock<spinlock> lock(m_mtx);

    m_queues[&lq] = &lqmtx;
}


void task_thief::unregister_queue(std::deque<task_info> &lq) {

    auto_lock<spinlock> lock(m_mtx);

    std::map< std::deque<task_info>*, spinlock* >::iterator i =
        m_queues.find(&lq);
    if(i == m_queues.end()) return;
    if(i == m_i) ++m_i;
    m_queues.erase(i);
}


void task_thief::steal_task(task_info &tinfo) {

    auto_lock<spinlock> lock(m_mtx);

    tinfo.tsrc = 0;
    tinfo.tsk = 0;

    if(m_queues.empty()) return;

    //  Round robin strategy for choosing a queue to steal from

    std::map< std::deque<task_info>*, spinlock* >::iterator iend = m_i, i = m_i;

    do {

        if(i == m_queues.end()) i = m_queues.begin();
        else ++i;

        if(i != m_queues.end()) {
            auto_lock<spinlock> lock(*i->second);
            if(!i->first->empty()) {
                tinfo = i->first->back();
                i->first->pop_back();
                break;
            }
        }

    } while(i != iend);
}


} // namespace libutil

