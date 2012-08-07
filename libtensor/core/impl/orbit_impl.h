#ifndef LIBTENSOR_ORBIT_IMPL_H
#define LIBTENSOR_ORBIT_IMPL_H

#include <vector>
#include "../orbit.h"

namespace libtensor {


template<size_t N, typename T>
const char *orbit<N, T>::k_clazz = "orbit<N, T>";


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx,
    bool compute_allowed) :

    m_bidims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(idx, m_bidims);
    build_orbit(sym, aidx0);
    abs_index<N>::get_index(m_orb.begin()->first, m_bidims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, size_t aidx,
    bool compute_allowed) :

    m_bidims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(aidx, m_bidims);
    build_orbit(sym, aidx0);
    abs_index<N>::get_index(m_orb.begin()->first, m_bidims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(const index<N> &idx) const {

    return get_transf(abs_index<N>::get_abs_index(idx, m_bidims));
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(size_t aidx) const {

#ifdef LIBTENSOR_DEBUG
    //  This will throw out_of_bounds if aidx is invalid
    abs_index<N>(aidx, m_bidims).get_index();
#endif // LIBTENSOR_DEBUG

    return get_transf(m_orb.find(aidx));
}


template<size_t N, typename T>
bool orbit<N, T>::contains(const index<N> &idx) const {

    return contains(abs_index<N>::get_abs_index(idx, m_bidims));
}


template<size_t N, typename T>
bool orbit<N, T>::contains(size_t aidx) const {

    return m_orb.find(aidx) != m_orb.end();
}


template<size_t N, typename T>
size_t orbit<N, T>::get_abs_index(const iterator &i) const {

    static const char *method = "get_abs_index(iterator&)";

#ifdef LIBTENSOR_DEBUG
    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }
#endif // LIBTENSOR_DEBUG

    return i->first;
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(const iterator &i) const {

    static const char *method = "get_transf(iterator&)";

#ifdef LIBTENSOR_DEBUG
    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }
#endif // LIBTENSOR_DEBUG

    return i->second;
}


template<size_t N, typename T>
void orbit<N, T>::build_orbit(const symmetry<N, T> &sym,
const abs_index<N> &aidx) {

    typedef index<N> index_type;

    //  Queue
    std::vector<index_type> qi;
    std::vector<tensor_transf_type> qt;

    //  Current indexes
    std::vector<index_type> ti;
    std::vector<tensor_transf_type> tt;

    qi.reserve(32);
    qt.reserve(32);
    ti.reserve(32);
    tt.reserve(32);

    tensor_transf_type tr1; // Identity transformation

    m_orb.insert(pair_type(aidx.get_abs_index(), tr1));

    qi.push_back(aidx.get_index());
    qt.push_back(tr1);

    while(!qi.empty()) {

        //  Pop an index from the queue
        index_type idx(qi.back());
        tensor_transf_type tr(qt.back());
        qi.pop_back();
        qt.pop_back();

        //  Apply all symmetry elements and save results
        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);
                if(m_allowed) m_allowed = elem.is_allowed(idx);
                ti.push_back(idx);
                tt.push_back(tr);
                elem.apply(ti.back(), tt.back());
            }
        }

        //  Discover any new indexes and push them to the queue
        for(size_t i = 0; i < ti.size(); i++) {
            size_t aidx = abs_index<N>::get_abs_index(ti[i], m_bidims);
            if(m_orb.insert(pair_type(aidx, tt[i])).second) {
                qi.push_back(ti[i]);
                qt.push_back(tt[i]);
            }
        }
        ti.clear();
        tt.clear();
    }

    //  At this point all the transformations are from the starter to targets.
    //  Need to translate them so the transformations are from the canonical
    //  block to the target blocks.

    //  Let 0 be the canonical block, s the starter block, i the target block.
    //  Then the orbit is currently populated with [i, T(s->i)]. This needs
    //  to be translated into [i, T(0->i)].
    //  T(i->0) = T(s->0) T(i->s) = T(s->0) Tinv(s->i)
    //  T(0->i) = Tinv(i->0) = [ T(s->0) Tinv(s->i) ]^(-1)

    tensor_transf_type tr0(m_orb.begin()->second); // T(s->0)
    for(orbit_map_iterator_type i = m_orb.begin(); i != m_orb.end(); ++i) {
        i->second.invert();
        i->second.transform(tr0);
        i->second.invert();
    }
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_IMPL_H
