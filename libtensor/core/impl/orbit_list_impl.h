#ifndef LIBTENSOR_ORBIT_LIST_IMPL_H
#define LIBTENSOR_ORBIT_LIST_IMPL_H

#include <cstring>
#include <libutil/threads/tls.h>
#include <libtensor/core/abs_index.h>
#include "../orbit_list.h"

namespace libtensor {


/** \brief Preallocated buffer for the orbit_list algorithm

    This is a per-thread buffer that helps reduce the load on malloc/free,
    one of the hotspots in the orbit_list algorithm.

    \ingroup libtensor_core
 **/
class orbit_list_buffer {
private:
    std::vector<char> m_v;

public:
    orbit_list_buffer() { }

    static std::vector<char> &get() {
        return libutil::tls<orbit_list_buffer>::get_instance().get().m_v;
    }
};


template<size_t N, typename T>
const char *orbit_list<N, T>::k_clazz = "orbit_list<N, T>";


template<size_t N, typename T>
orbit_list<N, T>::orbit_list(const symmetry<N, T> &sym) :

    m_dims(sym.get_bis().get_block_index_dims()) {

    orbit_list::start_timer();

    index<N> idx;
    size_t aidx = 0, n = m_dims.get_size();

    std::vector<char> &chk = orbit_list_buffer::get();
    if(chk.capacity() < n) chk.reserve(n);
    chk.resize(n, 0);
    ::memset(&chk[0], 0, n);

    const char *p0 = &chk[0];
    while(aidx < n) {
        const char *p = (const char*)::memchr(p0 + aidx, 0, n - aidx);
        if(p == 0) break;
        aidx = p - p0;
        abs_index<N>::get_index(aidx, m_dims, idx);
        if(mark_orbit(sym, idx, chk)) m_orb.push_back(aidx);
    }

    orbit_list::stop_timer();
}


template<size_t N, typename T>
bool orbit_list<N, T>::mark_orbit(const symmetry<N, T> &sym,
    const index<N> &idx, std::vector<char> &chk) {

    size_t aidx = abs_index<N>::get_abs_index(idx, m_dims);
    if(chk[aidx]) return true;

    bool allowed = true;
    chk[aidx] = 1;

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
