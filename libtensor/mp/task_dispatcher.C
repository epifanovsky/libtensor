#include "../exception.h"
#include "mp_exception.h"
#include "task_dispatcher.h"

namespace libtensor {


const char *task_dispatcher::k_clazz = "task_dispatcher";


task_dispatcher::task_dispatcher() :

    m_mp(false), m_ntasks(0), m_nwaiting(0) {

}


task_dispatcher::queue_id_t task_dispatcher::create_queue() {

    current_task_queue &ctq = tls<current_task_queue>::get_instance().get();
    task_queue *tq_parent = ctq.tq;
    if(tq_parent == 0) tq_parent = &m_root;
    task_queue *tq = new task_queue(tq_parent);

    {
        auto_lock lock(m_lock);
        return m_tqs.insert(m_tqs.end(), tq);
    }
}


void task_dispatcher::destroy_queue(queue_id_t &qid) {

    static const char *method = "destroy_queue(queue_id_t&)";

    current_task_queue &ctq = tls<current_task_queue>::get_instance().get();
    task_queue *tq = 0;

    {
        auto_lock lock(m_lock);

        if(qid == m_tqs.end()) {
            throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "qid");
        }

        tq = *qid;
        ctq.tq = tq->get_parent();
        m_tqs.erase(qid);
        qid = m_tqs.end();
    }

    delete tq;
}


void task_dispatcher::push_task(const queue_id_t &qid, task_i &task) {

    static const char *method = "push_task(const queue_id_t&, task_i&)";

    auto_lock lock(m_lock);

    if(qid == m_tqs.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "qid");
    }

    task_queue &q = **qid;
    q.push(&task);
    m_alarm.broadcast();
}


void task_dispatcher::wait_on_queue(const queue_id_t &qid) {

    cpu_pool cpus(1);
    wait_on_queue(qid, cpus);
}


void task_dispatcher::wait_on_queue(const queue_id_t &qid, cpu_pool &cpus) {

    task_queue &q = **qid;
    while(invoke_next(q, cpus));
    q.wait();
}


void task_dispatcher::set_off_alarm() {

    m_alarm.broadcast();
}


void task_dispatcher::wait_next() {

    if(!m_root.is_empty()) return;
    m_alarm.wait();
}


void task_dispatcher::invoke_next(cpu_pool &cpus) {

    invoke_next(m_root, cpus);
}


bool task_dispatcher::invoke_next(task_queue &tq, cpu_pool &cpus) {

    std::pair<task_queue*, task_i*> tt = tq.pop();
    if(tt.second == 0) return false;

    current_task_queue &ctq = tls<current_task_queue>::get_instance().get();
    ctq.tq = tt.first;

    try {
        tt.second->perform(cpus);
    } catch(exception &e) {
        tt.first->set_exception(e);
    } catch(...) {

    }

    tt.first->finished(*tt.second);
    return true;
}


} // namespace libtensor
