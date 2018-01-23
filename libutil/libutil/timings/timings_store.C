#include <algorithm>
#include <iomanip>
#include <libutil/threads/auto_lock.h>
#include "timings_store.h"

namespace libutil {


void timings_store_base::register_local(local_timings_store_base *lts) {

    auto_lock<mutex> lock(m_lock);

    m_lts.push_back(lts);
}


void timings_store_base::unregister_local(local_timings_store_base *lts) {

    auto_lock<mutex> lock(m_lock);

    std::vector<local_timings_store_base*>::iterator i =
        std::find(m_lts.begin(), m_lts.end(), lts);
    if(i != m_lts.end()) m_lts.erase(i);
}


void timings_store_base::reset() {

    auto_lock<mutex> lock(m_lock);

    for(std::vector<local_timings_store_base*>::iterator i = m_lts.begin();
        i != m_lts.end(); ++i) (*i)->reset();
}


size_t timings_store_base::get_ntimings() const {

    auto_lock<mutex> lock(m_lock);

    size_t n = 0;
    for(std::vector<local_timings_store_base*>::const_iterator i =
        m_lts.begin(); i != m_lts.end(); ++i) if(!(*i)->is_empty()) n++;
    return n;
}


time_diff_t timings_store_base::get_time(const std::string &id) const {

    std::map<std::string, timing_record> t;

    {
        auto_lock<mutex> lock(m_lock);
        for(std::vector<local_timings_store_base*>::const_iterator i =
            m_lts.begin(); i != m_lts.end(); ++i) (*i)->merge(t);
    }

    std::map<std::string, timing_record>::const_iterator i = t.find(id);
    if(i == t.end()) return time_diff_t();
    return i->second.m_total;
}


void timings_store_base::print(std::ostream& os) {

    std::map<std::string, timing_record> t;

    {
        auto_lock<mutex> lock(m_lock);
        for(std::vector<local_timings_store_base*>::iterator i = m_lts.begin();
            i != m_lts.end(); ++i) (*i)->merge(t);
    }

    for(std::map<std::string, timing_record>::const_iterator i = t.begin();
        i != t.end(); ++i) {

        os << "Execution of " << i->first << ": " << std::endl;
        os << "Calls: " << std::setw(10) << i->second.m_ncalls << ", "
            << i->second.m_total << std::endl;
    }
}


void timings_store_base::print_csv(std::ostream &os, char delim) {

    std::map<std::string, timing_record> t;

    {
        auto_lock<mutex> lock(m_lock);
        for(std::vector<local_timings_store_base*>::iterator i = m_lts.begin();
            i != m_lts.end(); ++i) (*i)->merge(t);
    }

    std::string comma(1, delim);
    for(std::map<std::string, timing_record>::const_iterator i = t.begin();
        i != t.end(); ++i) {

        os << i->first << comma << i->second.m_ncalls << comma;
        os << std::setprecision(2) << std::showpoint << std::fixed
            << i->second.m_total.user_time() << comma;
        os << std::setprecision(2) << std::showpoint << std::fixed
            << i->second.m_total.system_time() << comma;
        os << std::setprecision(2) << std::showpoint << std::fixed
            << i->second.m_total.wall_time() << std::endl;
    }
}


} // namespace libutil
