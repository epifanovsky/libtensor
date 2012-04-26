#ifndef LIBTENSOR_ORBIT_IMPL_H
#define LIBTENSOR_ORBIT_IMPL_H

#include "orbit.h"

namespace libtensor {


template<size_t N, typename T>
const char *orbit<N, T>::k_clazz = "orbit<N, T>";


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx) :

    m_bidims(sym.get_bis().get_block_index_dims()) {

    orbit<N, T>::start_timer();

    m_allowed = true;
    build_orbit(sym, idx);

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(const index<N> &idx) const {

    return get_transf(abs_index<N>(idx, m_bidims).get_abs_index());
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(size_t absidx) const {

    iterator i = m_orb.find(absidx);
    return get_transf(i);
}

template<size_t N, typename T>
bool orbit<N, T>::contains(const index<N> &idx) const {

    return contains(abs_index<N>(idx, m_bidims).get_abs_index());
}


template<size_t N, typename T>
bool orbit<N, T>::contains(size_t absidx) const {

    return m_orb.find(absidx) != m_orb.end();
}

template<size_t N, typename T>
size_t orbit<N, T>::get_abs_index(iterator &i) const {

    static const char *method = "get_abs_index(iterator&)";

#ifdef LIBTENSOR_DEBUG
    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i");
    }
#endif // LIBTENSOR_DEBUG

    return i->first;
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(iterator &i) const {

    static const char *method = "get_transf(iterator&)";

#ifdef LIBTENSOR_DEBUG
    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
            "i");
    }
#endif // LIBTENSOR_DEBUG

    return i->second;
}


template<size_t N, typename T>
void orbit<N, T>::build_orbit(const symmetry<N, T> &sym, const index<N> &idx) {

    std::vector< index<N> > qi;
    std::vector< tensor_transf<N, T> > qt;
    std::vector< index<N> > ti;
    std::vector< tensor_transf<N, T> > tt;

    qi.reserve(32);
    qt.reserve(32);
    ti.reserve(32);
    tt.reserve(32);

    abs_index<N> aidx0(idx, m_bidims);
    m_orb.insert(pair_t(aidx0.get_abs_index(), tensor_transf<N, T>()));

    qi.push_back(idx);
    qt.push_back(tensor_transf<N, T>());

    while(!qi.empty()) {

        index<N> idx1(qi.back());
        tensor_transf<N, T> tr1(qt.back());
        qi.pop_back();
        qt.pop_back();

        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); iset++) {

            const symmetry_element_set<N, T> &eset =
                sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator
                ielem = eset.begin(); ielem != eset.end();
                ielem++) {

                const symmetry_element_i<N, T> &elem =
                    eset.get_elem(ielem);
                m_allowed = m_allowed && elem.is_allowed(idx1);
                ti.push_back(idx1);
                tt.push_back(tr1);
                elem.apply(ti.back(), tt.back());
            }
        }
        for(size_t i = 0; i < ti.size(); i++) {
            abs_index<N> aidx(ti[i], m_bidims);
            std::pair<typename orbit_map_t::iterator, bool> it =
                    m_orb.insert(pair_t(aidx.get_abs_index(), tt[i]));
            if(it.second) {
                qi.push_back(ti[i]);
                qt.push_back(tt[i]);
            }
//            else if (tt[i].get_scalar_tr() !=
//                    it.first->second.get_scalar_tr()) {
//
//                m_allowed = false;
//            }
        }
        ti.clear();
        tt.clear();
    }

    tensor_transf<N, T> tr0(m_orb.begin()->second);
    tr0.invert();
    for(typename orbit_map_t::iterator i = m_orb.begin();
        i != m_orb.end(); i++) {
        i->second.transform(tr0);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_IMPL_H
