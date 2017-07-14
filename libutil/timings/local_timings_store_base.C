#include <libutil/exceptions/util_exceptions.h>
#include "local_timings_store_base.h"


namespace libutil {


local_timings_store_base::local_timings_store_base() {

    m_timers.reserve(32);
    for(size_t i = 0; i < 8; i++) m_timers.push_back(new timer());
}


local_timings_store_base::~local_timings_store_base() {

    reset();
    for(size_t i = 0; i < m_timers.size(); i++) delete m_timers[i];
}


void local_timings_store_base::start_timer(const std::string &name) {

    timer *t = 0;
    if(m_timers.empty()) {
        t = new timer();
    } else {
        t = m_timers.back();
        m_timers.pop_back();
    }
    m_incomplete.insert(incomplete_pair_type(name, t));
    t->start();
}


void local_timings_store_base::stop_timer(const std::string &name) {

    incomplete_map_type::iterator i = m_incomplete.find(name);
    if(i == m_incomplete.end()) {
        throw timings_exception("local_timings_store_base",
                "stop_timer(const std::string&)", __FILE__, __LINE__,
                "Unknown timer name.");
    }

    timer *t = i->second;
    t->stop();
    m_incomplete.erase(i);

    std::pair<complete_map_type::iterator, bool> r = m_complete.insert(
        complete_pair_type(name, timing_record(t->duration())));
    if(!r.second) r.first->second.add_call(t->duration());

    m_timers.push_back(t);
}


bool local_timings_store_base::is_empty() const {

    return m_complete.empty();
}


void local_timings_store_base::merge(std::map<std::string, timing_record> &t) {

    for(complete_map_type::iterator i = m_complete.begin();
        i != m_complete.end(); ++i) {

        std::pair<std::string, timing_record> p(i->first, i->second);
        std::pair<std::map<std::string, timing_record>::iterator, bool> r =
            t.insert(p);
        if(!r.second) r.first->second.add_calls(i->second);
    }
}


void local_timings_store_base::reset() {

    for(incomplete_map_type::iterator i = m_incomplete.begin();
        i != m_incomplete.end(); ++i) delete i->second;

    m_incomplete.clear();
    m_complete.clear();
}


} // namespace libutil
