#ifndef LIBTENSOR_COMBINED_ORBITS_IMPL_H
#define LIBTENSOR_COMBINED_ORBITS_IMPL_H

#include <cstring>
#include <libutil/threads/tls.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_block_index_space.h>
#include "../combined_orbits.h"

namespace libtensor {


class combined_orbits_buffer {
private:
    std::vector<size_t> m_o1, m_o2, m_o3, m_v1, m_v2, m_v3, m_w1, m_w2,
        m_q, m_q1, m_q2, m_t1, m_t2;

public:
    combined_orbits_buffer() {
        m_o1.reserve(32);
        m_o2.reserve(32);
        m_o3.reserve(32);
        m_v1.reserve(32);
        m_v2.reserve(32);
        m_v3.reserve(32);
        m_w1.reserve(64);
        m_w2.reserve(64);
        m_q.reserve(32);
        m_q1.reserve(64);
        m_q2.reserve(64);
        m_t1.reserve(64);
        m_t2.reserve(64);
    }

    static std::vector<size_t> &get_o1() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_o1;
    }

    static std::vector<size_t> &get_o2() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_o2;
    }

    static std::vector<size_t> &get_o3() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_o3;
    }

    static std::vector<size_t> &get_v1() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_v1;
    }

    static std::vector<size_t> &get_v2() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_v2;
    }

    static std::vector<size_t> &get_v3() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_v3;
    }

    static std::vector<size_t> &get_w1() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_w1;
    }

    static std::vector<size_t> &get_w2() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_w2;
    }

    static std::vector<size_t> &get_q() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_q;
    }

    static std::vector<size_t> &get_q1() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_q1;
    }

    static std::vector<size_t> &get_q2() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_q2;
    }

    static std::vector<size_t> &get_t1() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_t1;
    }

    static std::vector<size_t> &get_t2() {
        return libutil::tls<combined_orbits_buffer>::get_instance().get().m_t2;
    }

};


template<size_t N, typename T>
const char *combined_orbits<N, T>::k_clazz = "combined_orbits<N, T>";


template<size_t N, typename T>
combined_orbits<N, T>::combined_orbits(
    const symmetry<N, T> &sym1,
    const symmetry<N, T> &sym2,
    const symmetry<N, T> &sym3,
    size_t aidx) :

    m_dims(sym1.get_bis().get_block_index_dims()) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "combined_orbits()";

    if(!sym1.get_bis().equals(sym3.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "sym1,sym3");
    }
    if(!sym2.get_bis().equals(sym3.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "sym2,sym3");
    }
#endif // LIBTENSOR_DEBUG

    std::vector<size_t> &o1 = combined_orbits_buffer::get_o1();
    std::vector<size_t> &o2 = combined_orbits_buffer::get_o2();
    std::vector<size_t> &o3 = combined_orbits_buffer::get_o3();
    std::vector<size_t> &q1 = combined_orbits_buffer::get_q1();
    std::vector<size_t> &q2 = combined_orbits_buffer::get_q2();
    std::vector<size_t> &t1 = combined_orbits_buffer::get_t1();
    std::vector<size_t> &t2 = combined_orbits_buffer::get_t2();
    std::vector<size_t> &w1 = combined_orbits_buffer::get_w1();
    std::vector<size_t> &w2 = combined_orbits_buffer::get_w2();

    q1.clear();
    w1.clear();
    q2.clear();
    w2.clear();
    q1.push_back(aidx);
    w1.push_back(aidx);

    while(!q1.empty() || !q2.empty()) {

        if(!q1.empty()) {
            build_orbit(sym1, q1.back(), o1);
            q1.pop_back();
            t1.resize(o1.size() + w2.size());
            t2.resize(o1.size());
            typename std::vector<size_t>::iterator i1 = std::set_union(
                o1.begin(), o1.end(), w2.begin(), w2.end(), t1.begin());
            typename std::vector<size_t>::iterator i2 = std::set_difference(
                o1.begin(), o1.end(), w2.begin(), w2.end(), t2.begin());
            t1.resize(i1 - t1.begin());
            std::swap(w2, t1);
            q2.insert(q2.end(), t2.begin(), i2);
        }

        if(!q2.empty()) {
            build_orbit(sym2, q2.back(), o1);
            q2.pop_back();
            t1.resize(o1.size() + w1.size());
            t2.resize(o1.size());
            typename std::vector<size_t>::iterator i1 = std::set_union(
                o1.begin(), o1.end(), w1.begin(), w1.end(), t1.begin());
            typename std::vector<size_t>::iterator i2 = std::set_difference(
                o1.begin(), o1.end(), w1.begin(), w1.end(), t2.begin());
            t1.resize(i1 - t1.begin());
            std::swap(w1, t1);
            q1.insert(q1.end(), t2.begin(), i2);
        }
    }
    std::swap(w1, o1);

    while(!o1.empty()) {
        m_orb.push_back(o1[0]);
        o2.clear();
        build_orbit(sym3, o1[0], o2);
        o3.resize(o1.size());
        typename std::vector<size_t>::iterator i = std::set_difference(
            o1.begin(), o1.end(), o2.begin(), o2.end(), o3.begin());
        o3.resize(i - o3.begin());
        std::swap(o1, o3);
    }
}


template<size_t N, typename T>
void combined_orbits<N, T>::build_orbit(const symmetry<N, T> &sym, size_t aidx,
    std::vector<size_t> &orb) {

    std::vector<size_t> &v1 = combined_orbits_buffer::get_v1();
    std::vector<size_t> &v2 = combined_orbits_buffer::get_v2();
    std::vector<size_t> &v3 = combined_orbits_buffer::get_v3();
    std::vector<size_t> &q = combined_orbits_buffer::get_q();

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

        abs_index<N>::get_index(q.back(), m_dims, idx);
        q.pop_back();

        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);

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

    std::swap(v1, orb);

    v1.clear();
    v2.clear();
    v3.clear();
}


} // namespace libtensor

#endif // LIBTENSOR_COMBINED_ORBITS_IMPL_H
