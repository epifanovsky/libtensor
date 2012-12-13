#ifndef LIBTENSOR_SUBGROUP_ORBITS_IMPL_H
#define LIBTENSOR_SUBGROUP_ORBITS_IMPL_H

#include <cstring>
#include <libutil/threads/tls.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_block_index_space.h>
#include "../subgroup_orbits.h"

namespace libtensor {


class subgroup_orbits_buffer {
private:
    std::vector<size_t> m_o1, m_o2, m_o3, m_v1, m_v2, m_v3, m_q;

public:
    subgroup_orbits_buffer() {
        m_o1.reserve(32);
        m_o2.reserve(32);
        m_o3.reserve(32);
        m_v1.reserve(32);
        m_v2.reserve(32);
        m_v3.reserve(32);
        m_q.reserve(32);
    }

    static std::vector<size_t> &get_o1() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_o1;
    }

    static std::vector<size_t> &get_o2() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_o2;
    }

    static std::vector<size_t> &get_o3() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_o3;
    }

    static std::vector<size_t> &get_v1() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_v1;
    }

    static std::vector<size_t> &get_v2() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_v2;
    }

    static std::vector<size_t> &get_v3() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_v3;
    }

    static std::vector<size_t> &get_q() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_q;
    }

};


template<size_t N, typename T>
const char *subgroup_orbits<N, T>::k_clazz = "subgroup_orbits<N, T>";


template<size_t N, typename T>
subgroup_orbits<N, T>::subgroup_orbits(
    const symmetry<N, T> &sym1,
    const symmetry<N, T> &sym2,
    size_t aidx) :

    m_dims(sym1.get_bis().get_block_index_dims()), m_mdims(m_dims, true) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "subgroup_orbits()";

    block_index_space<N> bis1(sym1.get_bis()), bis2(sym2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    if(!bis1.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "sym1,sym2");
    }
#endif // LIBTENSOR_DEBUG

    std::vector<size_t> &o1 = subgroup_orbits_buffer::get_o1();
    std::vector<size_t> &o2 = subgroup_orbits_buffer::get_o2();
    std::vector<size_t> &o3 = subgroup_orbits_buffer::get_o3();

    o1.clear();
    o2.clear();
    o3.clear();

    build_orbit(sym1, aidx, o1);

    while(!o1.empty()) {
        m_orb.push_back(o1[0]);
        o2.clear();
        build_orbit(sym2, o1[0], o2);
        o3.resize(o1.size());
        typename std::vector<size_t>::iterator i = std::set_difference(
            o1.begin(), o1.end(), o2.begin(), o2.end(), o3.begin());
        o3.resize(i - o3.begin());
        std::swap(o1, o3);
    }
}


template<size_t N, typename T>
void subgroup_orbits<N, T>::build_orbit(const symmetry<N, T> &sym, size_t aidx,
    std::vector<size_t> &orb) {

    std::vector<size_t> &v1 = subgroup_orbits_buffer::get_v1();
    std::vector<size_t> &v2 = subgroup_orbits_buffer::get_v2();
    std::vector<size_t> &v3 = subgroup_orbits_buffer::get_v3();
    std::vector<size_t> &q = subgroup_orbits_buffer::get_q();

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

#endif // LIBTENSOR_SUBGROUP_ORBITS_IMPL_H
