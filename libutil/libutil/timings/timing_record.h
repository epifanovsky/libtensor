#ifndef LIBUTIL_TIMING_RECORD_H
#define LIBUTIL_TIMING_RECORD_H

#include "timer.h"

namespace libutil {


struct timing_record {

    time_diff_t m_total;
    size_t m_ncalls;

    timing_record(const time_diff_t &t) : m_total(t), m_ncalls(1) {

    }

    void add_call(const time_diff_t &t) {
        m_ncalls++;
        m_total += t;
    }

    void add_calls(const timing_record &other) {
        m_ncalls += other.m_ncalls;
        m_total += other.m_total;
    }

};


} // namespace libutil

#endif // LIBUTIL_TIMING_RECORD_H
