#ifndef LIBTENSOR_SHORT_ORBIT_IMPL_H
#define LIBTENSOR_SHORT_ORBIT_IMPL_H

#include <algorithm>
#include <vector>
#include "../short_orbit.h"

namespace libtensor {


class short_orbit_buffer {
private:
    std::vector<size_t> m_v1, m_v2, m_v3, m_q;

public:
    short_orbit_buffer() {
        m_v1.reserve(32);
        m_v2.reserve(32);
        m_v3.reserve(32);
        m_q.reserve(32);
    }

    static std::vector<size_t> &get_v1() {
        return libutil::tls<short_orbit_buffer>::get_instance().get().m_v1;
    }

    static std::vector<size_t> &get_v2() {
        return libutil::tls<short_orbit_buffer>::get_instance().get().m_v2;
    }

    static std::vector<size_t> &get_v3() {
        return libutil::tls<short_orbit_buffer>::get_instance().get().m_v3;
    }

    static std::vector<size_t> &get_q() {
        return libutil::tls<short_orbit_buffer>::get_instance().get().m_q;
    }

};


template<size_t N, typename T>
const char *short_orbit<N, T>::k_clazz = "short_orbit<N, T>";


template<size_t N, typename T>
short_orbit<N, T>::short_orbit(const symmetry<N, T> &sym, const index<N> &idx,
    bool compute_allowed) :

    m_dims(sym.get_bis().get_block_index_dims()), m_mdims(m_dims, true),
    m_allowed(true) {

    short_orbit::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    find_cindex(sym, abs_index<N>::get_abs_index(idx, m_dims));
    abs_index<N>::get_index(m_acidx, m_mdims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    short_orbit::stop_timer();
}


template<size_t N, typename T>
short_orbit<N, T>::short_orbit(const symmetry<N, T> &sym, size_t aidx,
    bool compute_allowed) :

    m_dims(sym.get_bis().get_block_index_dims()), m_mdims(m_dims, true),
    m_allowed(true) {

    short_orbit::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    find_cindex(sym, aidx);
    abs_index<N>::get_index(m_acidx, m_mdims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    short_orbit::stop_timer();
}


template<size_t N, typename T>
void short_orbit<N, T>::find_cindex(const symmetry<N, T> &sym, size_t aidx) {

    std::vector<size_t> &v1 = short_orbit_buffer::get_v1();
    std::vector<size_t> &v2 = short_orbit_buffer::get_v2();
    std::vector<size_t> &v3 = short_orbit_buffer::get_v3();
    std::vector<size_t> &q = short_orbit_buffer::get_q();

    //  v1 -- visited indexes, sorted
    //  v2 -- newly discovered indexes, unsorted
    //  v3 -- generated from v2 by sorting its unique entries
    //  q  -- queue

    v1.clear();
    q.clear();
    q.push_back(aidx);
    v1.push_back(aidx);

    index<N> idx;
    while(!q.empty()) {

        v2.clear();
        v3.clear();

        abs_index<N>::get_index(q.back(), m_mdims, idx);
        q.pop_back();

        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);

                if(m_allowed) m_allowed = elem.is_allowed(idx);
                index<N> idx2(idx);
                elem.apply(idx2);
                size_t aidx2 = abs_index<N>::get_abs_index(idx2, m_dims);
                if(!std::binary_search(v1.begin(), v1.end(), aidx2)) {
                    v2.push_back(aidx2);
                }
            }
        }
        //  Sort v2 & remove all duplicates
        std::sort(v2.begin(), v2.end());
        v2.resize(std::unique(v2.begin(), v2.end()) - v2.begin());
        if(!v2.empty()) {
            //  v2 now contains all new indexes to visit
            //  Put the in the queue
            q.insert(q.end(), v2.begin(), v2.end());
            v3.resize(v1.size() + v2.size());
            std::merge(v1.begin(), v1.end(), v2.begin(), v2.end(), v3.begin());
            std::swap(v1, v3);
        }
    }

    m_acidx = v1[0];

    v1.clear();
    v2.clear();
    v3.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_SHORT_ORBIT_IMPL_H
