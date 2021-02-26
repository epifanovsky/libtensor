#ifndef LIBTENSOR_SPLIT_POINTS_H
#define LIBTENSOR_SPLIT_POINTS_H

#include <vector>
#include "../defs.h"
#include "../exception.h"

namespace libtensor {


/** \brief Contains the position of %index space splitting points

    \ingroup libtensor_core
 **/
class split_points {
private:
    std::vector<size_t> m_points; //!< Positions of split points

public:
    bool add(size_t pos);
    bool equals(const split_points &sp) const;
    size_t get_num_points() const;
    size_t operator[](size_t i) const;
};


inline bool split_points::add(size_t pos) {

    bool inc = false;
    std::vector<size_t>::iterator isp = m_points.begin();
    while(isp != m_points.end()) {

        size_t curpos = *isp;
        if(curpos == pos) break;
        if(curpos > pos) {
            isp = m_points.insert(isp, pos);
            inc = true;
            break;
        }
        isp++;
    }
    if(isp == m_points.end()) {
        m_points.push_back(pos);
        inc = true;
    }
    return inc;
}


inline bool split_points::equals(const split_points &sp) const {

    size_t sz = m_points.size();
    if(sp.m_points.size() != sz) return false;
    for(size_t i = 0; i < sz; i++)
        if(m_points[i] != sp.m_points[i]) return false;
    return true;
}


inline size_t split_points::get_num_points() const {

    return m_points.size();
}


inline size_t split_points::operator[](size_t i) const {

#ifdef LIBTENSOR_DEBUG
    if(i >= m_points.size()) {
        throw out_of_bounds(g_ns, "split_points", "operator[](size_t)",
            __FILE__, __LINE__, "Point number is out of bounds.");
    }
#endif // LIBTENSOR_DEBUG
    return m_points[i];
}


} // namespace libtensor

#endif // LIBTENSOR_SPLIT_POINTS_H
