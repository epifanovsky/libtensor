#include <algorithm>
#include <libutil/threads/auto_lock.h>
#include "task_source.h"

namespace libutil {


task_source::task_source(task_source *parent, task_iterator_i &ti,
    task_observer_i &to) :

    m_parent(parent), m_exc(0), m_ti(ti), m_to(to), m_npending(0),
    m_nrunning(0) {

    if(m_parent) m_parent->add_child(this);
}


task_source::~task_source() {

    if(m_parent) m_parent->remove_child(this);
}


void task_source::wait() {

    while(!is_alldone()) m_alldone.wait();
}


void task_source::rethrow_exceptions() {

    if(m_exc) {
        try {
            m_exc->rethrow();
        } catch(...) {
            delete m_exc;
            m_exc = 0;
            throw;
        }
    }
}

task_source *task_source::get_current() {

    auto_lock<mutex> lock(m_mtx);

    for(std::list<task_source*>::iterator i = m_children.begin();
            i != m_children.end(); ++i) {
        task_source *src = (*i)->get_current();
        if(src) return src;
    }

    if(m_ti.has_more()) return this;
    return 0;
}


task_i *task_source::extract_task() {

    auto_lock<mutex> lock(m_mtx);

    task_i *t = 0;
    if(m_ti.has_more()) {
        t = m_ti.get_next();
        if(t) m_npending++;
    }
    return t;
}


void task_source::notify_start_task(task_i *t) {

    if(t == 0) return;

    {
        auto_lock<mutex> lock(m_mtx);
        m_npending--;
        m_nrunning++;
    }

    m_to.notify_start_task(t);
}


void task_source::notify_finish_task(task_i *t) {

    if(t == 0) return;

    m_to.notify_finish_task(t);

    {
        auto_lock<mutex> lock(m_mtx);
        m_nrunning--;
        if(is_alldone_unsafe()) m_alldone.signal();
    }
}


void task_source::notify_exception(task_i *t, const rethrowable_i &e) {

    {
        auto_lock<mutex> lock(m_mtx);
        if(m_exc == 0) m_exc = e.clone();
    }
}


void task_source::add_child(task_source *ts) {

    auto_lock<mutex> lock(m_mtx);
    m_children.push_back(ts);
}


void task_source::remove_child(task_source *ts) {

    auto_lock<mutex> lock(m_mtx);
    std::list<task_source*>::iterator i =
        std::find(m_children.begin(), m_children.end(), ts);
    if(i != m_children.end()) m_children.erase(i);
    if(is_alldone_unsafe()) m_alldone.signal();
}


bool task_source::is_alldone() {

    auto_lock<mutex> lock(m_mtx);
    return is_alldone_unsafe();
}


bool task_source::is_alldone_unsafe() {

    return (m_npending == 0 && m_nrunning == 0) &&
        m_children.empty() && !m_ti.has_more();
}


} // namespace libutil

