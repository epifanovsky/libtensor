#include "task_queue.h"
#include "mp_exception.h"

namespace libtensor {


const char *task_queue::k_clazz = "task_queue";


task_queue::task_queue(task_queue *parent) :

    m_queued(0), m_inprogress(0), m_parent(parent), m_exc(0) {

    if(m_parent) m_parent->add_child(this);
}


task_queue::~task_queue() {

    delete m_exc; m_exc = 0;
    if(m_parent) m_parent->remove_child(this);
}


bool task_queue::is_empty() const {

    {
        auto_lock lock(m_lock);

        if(m_queued > 0) return false;

        std::set<task_queue*>::const_iterator i = m_children.begin();
        for(; i != m_children.end(); ++i) {
            if(!(*i)->is_empty()) return false;
        }
    }
    return true;
}


void task_queue::push(task_i *task) {

    static const char *method = "push(task_i*)";

#ifdef LIBTENSOR_DEBUG
    if(task == 0) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "task");
    }
#endif // LIBTENSOR_DEBUG
    {
        auto_lock lock(m_lock);
        m_q.push_back(task);
        m_queued++;
    }
}


std::pair<task_queue*, task_i*> task_queue::pop() {

    static const char *method = "pop()";

    std::pair<task_queue*, task_i*> tt(0, 0);

    {
        auto_lock lock(m_lock);

        //  Scan through the children first
        std::set<task_queue*>::iterator i = m_children.begin();
        for(; i != m_children.end(); ++i) {
            tt = (*i)->pop();
            if(tt.second) return tt;
        }

        tt.first = this;

        //  Then try the current queue
        if(!m_q.empty()) {
            tt.second = m_q.front();
            m_q.pop_front();
            m_queued--;
            m_inprogress++;
        } else {
            tt.second = 0;
        }
    }
    return tt;
}


void task_queue::finished(task_i &task) {

    bool empty = false;
    {
        auto_lock lock(m_lock);
        m_inprogress--;
        empty = (m_queued == 0 && m_inprogress == 0);
    }
    if(empty) m_sig.broadcast();
}


void task_queue::set_exception(exception &exc) {

    auto_lock lock(m_lock);

    if(!m_exc) m_exc = exc.clone();
}


void task_queue::wait() {

    while(true) {
        {
            auto_lock lock(m_lock);
            if(m_queued == 0 && m_inprogress == 0) break;
        }
        m_sig.wait();
    }

    {
        auto_lock lock(m_lock);

        if(m_exc) {
            try {
                m_exc->rethrow();
            } catch(exception &e) {
                delete m_exc;
                m_exc = 0;
                throw;
            }
        }
    }
}


void task_queue::add_child(task_queue *q) {

    auto_lock lock(m_lock);
    m_children.insert(q);
}


void task_queue::remove_child(task_queue *q) {

    auto_lock lock(m_lock);
    m_children.erase(q);
}


} // namespace libtensor
