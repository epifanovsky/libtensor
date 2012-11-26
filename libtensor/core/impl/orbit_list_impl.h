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
    std::vector<size_t> m_q;

public:
    orbit_list_buffer() {
        m_q.reserve(32);
    }

    static std::vector<char> &get_v() {
        return libutil::tls<orbit_list_buffer>::get_instance().get().m_v;
    }

    static std::vector<size_t> &get_q() {
        return libutil::tls<orbit_list_buffer>::get_instance().get().m_q;
    }

};


template<size_t N, typename T>
const char *orbit_list<N, T>::k_clazz = "orbit_list<N, T>";


template<size_t N, typename T>
orbit_list<N, T>::orbit_list(const symmetry<N, T> &sym) :

    m_dims(sym.get_bis().get_block_index_dims()),
    m_mdims(m_dims, true) {

    orbit_list::start_timer();

    index<N> idx;
    size_t aidx = 0, n = m_dims.get_size();

    std::vector<char> &chk = orbit_list_buffer::get_v();
    if(chk.capacity() < n) chk.reserve(n);
    chk.resize(n, 0);
    ::memset(&chk[0], 0, n);

    const char *p0 = &chk[0];
    while(aidx < n) {
        const char *p = (const char*)::memchr(p0 + aidx, 0, n - aidx);
        if(p == 0) break;
        aidx = p - p0;
        if(mark_orbit(sym, aidx, chk)) m_orb.push_back(aidx);
    }

    orbit_list::stop_timer();
}


template<size_t N, typename T>
bool orbit_list<N, T>::mark_orbit(const symmetry<N, T> &sym, size_t aidx0,
    std::vector<char> &chk) {

    std::vector<size_t> &q = orbit_list_buffer::get_q();

    bool allowed = true;
    q.clear();
    q.push_back(aidx0);
    chk[aidx0] = 1;

    index<N> idx;
    while(!q.empty()) {

        size_t aidx = q.back();
        q.pop_back();
        abs_index<N>::get_index(aidx, m_mdims, idx);

        for(typename symmetry<N, T>::iterator iset = sym.begin();
            iset != sym.end(); ++iset) {

            const symmetry_element_set<N, T> &eset = sym.get_subset(iset);
            for(typename symmetry_element_set<N, T>::const_iterator ielem =
                eset.begin(); ielem != eset.end(); ++ielem) {

                const symmetry_element_i<N, T> &elem = eset.get_elem(ielem);
                if(allowed) allowed = elem.is_allowed(idx);
                index<N> idx2(idx);
                elem.apply(idx2);
                size_t aidx2 = abs_index<N>::get_abs_index(idx2, m_dims);
                if(chk[aidx2] == 0) {
                    q.push_back(aidx2);
                    chk[aidx2] = 1;
                }
            }
        }
    }

    return allowed;
}


} // namespace libtensor

#endif // LIBTENSOR_ORBIT_LIST_IMPL_H
