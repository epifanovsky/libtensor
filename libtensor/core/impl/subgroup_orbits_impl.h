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
    std::vector<char> m_v;
    std::vector<size_t> m_q;

public:
    subgroup_orbits_buffer() {
        m_q.reserve(32);
    }

    static std::vector<char> &get_v() {
        return libutil::tls<subgroup_orbits_buffer>::get_instance().get().m_v;
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

    m_dims(sym1.get_bis().get_block_index_dims()) {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "subgroup_orbits()";

    if(!sym1.get_bis().equals(sym2.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "sym1,sym2");
    }
#endif // LIBTENSOR_DEBUG

    index<N> idx;
    size_t n = m_dims.get_size();

    std::vector<char> &chk = subgroup_orbits_buffer::get_v();
    if(chk.capacity() < n) chk.reserve(n);
    chk.resize(n, 0);
    ::memset(&chk[0], 0, n);

    mark_orbit(sym1, aidx, chk, 1);
    const char *p0 = &chk[0];
    size_t aidx1 = aidx;
    while(aidx1 < n) {
        const char *p = (const char*)::memchr(p0 + aidx1, 1, n - aidx1);
        if(p == 0) break;
        aidx1 = p - p0;
        mark_orbit(sym2, aidx1, chk, 2);
        m_orb.push_back(aidx1);
    }
}


template<size_t N, typename T>
void subgroup_orbits<N, T>::mark_orbit(const symmetry<N, T> &sym, size_t aidx0,
    std::vector<char> &chk, char v) {

    std::vector<size_t> &q = subgroup_orbits_buffer::get_q();

    q.clear();
    q.push_back(aidx0);
    chk[aidx0] = v;

    index<N> idx;
    while(!q.empty()) {

        size_t aidx = q.back();
        q.pop_back();
        abs_index<N>::get_index(aidx, m_dims, idx);

        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);
                index<N> idx2(idx);
                elem.apply(idx2);
                size_t aidx2 = abs_index<N>::get_abs_index(idx2, m_dims);
                if(chk[aidx2] != v) {
                    q.push_back(aidx2);
                    chk[aidx2] = v;
                }
            }
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SUBGROUP_ORBITS_IMPL_H
