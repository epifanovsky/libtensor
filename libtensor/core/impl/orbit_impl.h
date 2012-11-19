#ifndef LIBTENSOR_ORBIT_IMPL_H
#define LIBTENSOR_ORBIT_IMPL_H

#include <cstring>
#include <algorithm>
#include <libutil/threads/tls.h>
#include "../orbit.h"

namespace libtensor {


namespace {


template<size_t N, typename T>
struct itr_pair_comp_less {
    bool operator()(
        const typename orbit<N, T>::itr_pair_type &p1,
        const typename orbit<N, T>::itr_pair_type &p2) {
        return p1.first < p2.first;
    }
};


template<size_t N, typename T>
struct itr_pair_comp_equal {
    bool operator()(
        const typename orbit<N, T>::itr_pair_type &p1,
        const typename orbit<N, T>::itr_pair_type &p2) {
        return p1.first == p2.first;
    }
};


template<size_t N, typename T>
class orbit_buffer {
public:
    typedef typename orbit<N, T>::itr_pair_type itr_pair_type;

private:
    std::vector< index<N> > m_qi, m_ti;
    std::vector< tensor_transf<N, T> > m_qt, m_tt;
    std::vector< itr_pair_type > m_can1, m_can2;

public:
    orbit_buffer() {
        m_qi.reserve(32);
        m_ti.reserve(32);
        m_qt.reserve(32);
        m_tt.reserve(32);
        m_can1.reserve(32);
        m_can2.reserve(32);
    }

    static std::vector< index<N> > &get_qi() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_qi;
    }

    static std::vector< index<N> > &get_ti() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_ti;
    }

    static std::vector< tensor_transf<N, T> >&get_qt() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_qt;
    }

    static std::vector< tensor_transf<N, T> > &get_tt() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_tt;
    }

    static std::vector<itr_pair_type> &get_can1() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_can1;
    }

    static std::vector<itr_pair_type> &get_can2() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_can2;
    }

};


} // unnamed namespace


template<size_t N, typename T>
const char *orbit<N, T>::k_clazz = "orbit<N, T>";


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx,
    bool compute_allowed) :

    m_dims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(idx, m_dims);
    build_orbit(sym, aidx0);
    abs_index<N>::get_index(m_orb.begin()->first, m_dims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, size_t aidx,
    bool compute_allowed) :

    m_dims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(aidx, m_dims);
    build_orbit(sym, aidx0);
    abs_index<N>::get_index(m_orb.begin()->first, m_dims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(const index<N> &idx) const {

    return get_transf(abs_index<N>::get_abs_index(idx, m_dims));
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(size_t aidx) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_transf(size_t)";

    //  This will throw out_of_bounds if aidx is invalid
    abs_index<N>(aidx, m_dims).get_index();
#endif // LIBTENSOR_DEBUG

    itr_pair_comp_less<N, T> comp;
    itr_pair_type aidxp(aidx, 0);
    iterator i = std::lower_bound(m_orb.begin(), m_orb.end(), aidxp, comp);
#ifdef LIBTENSOR_DEBUG
    if(i == m_orb.end() || i->first != aidx) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "aidx");
    }
#endif // LIBTENSOR_DEBUG
    return m_tr[i->second];
}


template<size_t N, typename T>
bool orbit<N, T>::contains(const index<N> &idx) const {

    return contains(abs_index<N>::get_abs_index(idx, m_dims));
}


template<size_t N, typename T>
bool orbit<N, T>::contains(size_t aidx) const {

    itr_pair_comp_less<N, T> comp;
    itr_pair_type aidxp(aidx, 0);
    return std::binary_search(m_orb.begin(), m_orb.end(), aidxp, comp);
}


template<size_t N, typename T>
size_t orbit<N, T>::get_abs_index(const iterator &i) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_abs_index(iterator&)";

    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }
#endif // LIBTENSOR_DEBUG

    return i->first;
}


template<size_t N, typename T>
const tensor_transf<N, T> &orbit<N, T>::get_transf(const iterator &i) const {

#ifdef LIBTENSOR_DEBUG
    static const char *method = "get_transf(iterator&)";

    if(i == m_orb.end()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "i");
    }
#endif // LIBTENSOR_DEBUG

    return m_tr[i->second];
}


template<size_t N, typename T>
void orbit<N, T>::build_orbit(const symmetry<N, T> &sym,
    const abs_index<N> &aidx) {

    typedef index<N> index_type;
    itr_pair_comp_less<N, T> comp;
    itr_pair_comp_equal<N, T> compeq;

    //  Queue
    std::vector<index_type> &qi = orbit_buffer<N, T>::get_qi();
    std::vector<tensor_transf_type> &qt = orbit_buffer<N, T>::get_qt();
    qi.clear();
    qt.clear();

    //  Current indexes
    std::vector<itr_pair_type> &can1 = orbit_buffer<N, T>::get_can1();
    std::vector<itr_pair_type> &can2 = orbit_buffer<N, T>::get_can2();
    std::vector<index_type> &ti = orbit_buffer<N, T>::get_ti();
    std::vector<tensor_transf_type> &tt = orbit_buffer<N, T>::get_tt();
    can1.clear();
    can2.clear();
    ti.clear();
    tt.clear();

    tensor_transf_type tr1; // Identity transformation

    m_tr.reserve(32);
    m_orb.reserve(32);
    m_tr.push_back(tr1);
    m_orb.push_back(std::make_pair(aidx.get_abs_index(), 0));

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
        //  1. Find candidates for addition
        for(size_t i = 0; i < ti.size(); i++) {
            size_t aidx1 = abs_index<N>::get_abs_index(ti[i], m_dims);
            itr_pair_type aidxp(aidx1, i);
            if(!std::binary_search(m_orb.begin(), m_orb.end(), aidxp, comp)) {
                can1.push_back(aidxp);
            }
        }
        std::sort(can1.begin(), can1.end(), comp);
        can1.resize(std::unique(can1.begin(), can1.end(), compeq) -
            can1.begin());
        //  2. Queue new unique indexes
        for(size_t i = 0; i < can1.size(); i++) {
            size_t aidx1 = can1[i].first;
            index<N> idx1;
            abs_index<N>::get_index(aidx1, m_dims, idx1);
            size_t j = m_tr.size();
            m_tr.push_back(tt[can1[i].second]);
            can2.push_back(std::make_pair(aidx1, j));
            qi.push_back(idx1);
            qt.push_back(m_tr[j]);
        }
        ti.clear();
        tt.clear();
        //  3. Merge new indexes into orbit
        can1.resize(m_orb.size() + can2.size());
        std::merge(m_orb.begin(), m_orb.end(), can2.begin(), can2.end(),
            can1.begin(), comp);
        m_orb.swap(can1);
        can1.clear();
        can2.clear();
    }

    //  At this point all the transformations are from the starter to targets.
    //  Need to translate them so the transformations are from the canonical
    //  block to the target blocks.

    //  Let 0 be the canonical block, s the starter block, i the target block.
    //  Then the orbit is currently populated with [i, T(s->i)]. This needs
    //  to be translated into [i, T(0->i)].
    //  T(i->0) = T(s->0) T(i->s) = T(s->0) Tinv(s->i)
    //  T(0->i) = Tinv(i->0) = [ T(s->0) Tinv(s->i) ]^(-1)

    tensor_transf_type tr0(m_tr[m_orb[0].second]); // T(s->0)
    for(typename std::vector<itr_pair_type>::iterator i = m_orb.begin();
        i != m_orb.end(); ++i) {
        m_tr[i->second].invert();
        m_tr[i->second].transform(tr0);
        m_tr[i->second].invert();
    }
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_IMPL_H
