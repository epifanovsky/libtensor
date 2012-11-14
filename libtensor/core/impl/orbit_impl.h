#ifndef LIBTENSOR_ORBIT_IMPL_H
#define LIBTENSOR_ORBIT_IMPL_H

#include <cstring>
#include <vector>
#include <libutil/threads/tls.h>
#include "../orbit.h"

namespace libtensor {


template<size_t N, typename T>
class orbit_buffer {
private:
    std::vector< index<N> > m_qi, m_ti;
    std::vector< tensor_transf<N, T> > m_qt, m_tt;
    std::vector<char> m_v;
    std::vector<size_t> m_q;

public:
    orbit_buffer() {
        m_qi.reserve(32);
        m_ti.reserve(32);
        m_qt.reserve(32);
        m_tt.reserve(32);
        m_q.reserve(32);
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

    static std::vector<char> &get_v() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_v;
    }

    static std::vector<size_t> &get_q() {
        return libutil::tls< orbit_buffer<N, T> >::get_instance().get().m_q;
    }

};


template<size_t N, typename T>
const char *orbit<N, T>::k_clazz = "orbit<N, T>";


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, const index<N> &idx,
    bool compute_allowed, bool cindex_only) :

    m_bidims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(idx, m_bidims);
    if(cindex_only) find_cindex(sym, aidx0);
    else build_orbit(sym, aidx0);
    abs_index<N>::get_index(m_orb.begin()->first, m_bidims, m_cidx);

    if(!compute_allowed) m_allowed = true;

    orbit<N, T>::stop_timer();
}


template<size_t N, typename T>
orbit<N, T>::orbit(const symmetry<N, T> &sym, size_t aidx,
    bool compute_allowed, bool cindex_only) :

    m_bidims(sym.get_bis().get_block_index_dims()), m_allowed(true) {

    orbit<N, T>::start_timer();

    //  Setting m_allowed to false will disable further calls to
    //  symmetry_element_i::is_allowed
    if(!compute_allowed) m_allowed = false;

    abs_index<N> aidx0(aidx, m_bidims);
    if(cindex_only) find_cindex(sym, aidx0);
    else build_orbit(sym, aidx0);
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
    std::vector<index_type> &qi = orbit_buffer<N, T>::get_qi();
    std::vector<tensor_transf_type> &qt = orbit_buffer<N, T>::get_qt();
    qi.clear();
    qt.clear();

    //  Current indexes
    std::vector<index_type> &ti = orbit_buffer<N, T>::get_ti();
    std::vector<tensor_transf_type> &tt = orbit_buffer<N, T>::get_tt();
    ti.clear();
    tt.clear();

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


template<size_t N, typename T>
void orbit<N, T>::find_cindex(const symmetry<N, T> &sym,
    const abs_index<N> &aidx) {

    typedef index<N> index_type;

    size_t n = m_bidims.get_size();
    std::vector<char> &chk = orbit_buffer<N, T>::get_v();
    if(chk.capacity() < n) chk.reserve(n);
    chk.resize(n, 0);
    ::memset(&chk[0], 0, n);

    //  Queue
    std::vector<index_type> &qi = orbit_buffer<N, T>::get_qi();
    qi.clear();

    //  Current indexes
    std::vector<index_type> &ti = orbit_buffer<N, T>::get_ti();
    ti.clear();

    size_t acidx = aidx.get_abs_index(); // Current smallest index in orbit

    qi.push_back(aidx.get_index());

    while(!qi.empty()) {

        //  Pop an index from the queue
        index_type idx(qi.back());
        qi.pop_back();

        //  Apply all symmetry elements and save results
        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);
                ti.push_back(idx);
                elem.apply(ti.back());
            }
        }

        //  Discover any new indexes and push them to the queue
        for(size_t i = 0; i < ti.size(); i++) {
            size_t aidx = abs_index<N>::get_abs_index(ti[i], m_bidims);
            if(chk[aidx] == 0) {
                qi.push_back(ti[i]);
                if(aidx < acidx) acidx = aidx;
                chk[aidx] = 1;
            }
        }
        ti.clear();
    }

    m_orb.insert(pair_type(acidx, tensor_transf_type()));
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_IMPL_H
