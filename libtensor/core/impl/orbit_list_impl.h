#ifndef LIBTENSOR_ORBIT_LIST_IMPL_H
#define LIBTENSOR_ORBIT_LIST_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../orbit_list.h"

namespace libtensor {


template<size_t N, typename T>
const char *orbit_list<N, T>::k_clazz = "orbit_list<N, T>";


template<size_t N, typename T>
orbit_list<N, T>::orbit_list(const symmetry<N, T> &sym) :
    m_dims(sym.get_bis().get_block_index_dims()) {

    orbit_list<N, T>::start_timer();

    std::vector<char> chk(m_dims.get_size(), 0);
    abs_index<N> aidx(m_dims);
    do {
        size_t absidx = aidx.get_abs_index();
        if(chk[absidx] == 0) {
            if(mark_orbit(sym, aidx.get_index(), chk)) {
                m_orb.insert(std::make_pair(absidx, aidx.get_index()));
            }
        }
    } while(aidx.inc());

    orbit_list<N, T>::stop_timer();
}


template<size_t N, typename T>
size_t orbit_list<N, T>::get_size() const {

    return m_orb.size();
}


template<size_t N, typename T>
bool orbit_list<N, T>::contains(const index<N> &idx) const {

    return contains(abs_index<N>::get_abs_index(idx, m_dims));
}


template<size_t N, typename T>
bool orbit_list<N, T>::contains(size_t absidx) const {

    return m_orb.find(absidx) != m_orb.end();
}


template<size_t N, typename T>
typename orbit_list<N, T>::iterator orbit_list<N, T>::begin() const {

    return m_orb.begin();
}


template<size_t N, typename T>
typename orbit_list<N, T>::iterator orbit_list<N, T>::end() const {

    return m_orb.end();
}


template<size_t N, typename T>
size_t orbit_list<N, T>::get_abs_index(iterator &i) const {

    return i->first;
}


template<size_t N, typename T>
const index<N> &orbit_list<N, T>::get_index(iterator &i) const {

    return i->second;
}


template<size_t N, typename T>
bool orbit_list<N, T>::mark_orbit(const symmetry<N, T> &sym,
    const index<N> &idx, std::vector<char> &chk) {

    size_t absidx = abs_index<N>::get_abs_index(idx, m_dims);
    if(chk[absidx]) return true;

    bool allowed = true;
    chk[absidx] = 1;

    for(typename symmetry<N, T>::iterator iset = sym.begin();
        iset != sym.end(); ++iset) {

        const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
        for(typename symmetry_element_set<N, T>::const_iterator ielem =
            eset.begin(); ielem != eset.end(); ++ielem) {

            const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);
            if(allowed) allowed = elem.is_allowed(idx);
            index<N> idx2(idx);
            elem.apply(idx2);
            bool allowed2 = mark_orbit(sym, idx2, chk);
            allowed = allowed && allowed2;
        }
    }
    return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_IMPL_H
