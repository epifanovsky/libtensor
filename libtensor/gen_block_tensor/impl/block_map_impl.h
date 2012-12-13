#ifndef LIBTENSOR_BLOCK_MAP_IMPL_H
#define LIBTENSOR_BLOCK_MAP_IMPL_H

#include <libtensor/core/abs_index.h>
#include "../block_map.h"

namespace libtensor {


template<size_t N, typename BtTraits>
const char *block_map<N, BtTraits>::k_clazz = "block_map<N, BtTraits>";


template<size_t N, typename BtTraits>
block_map<N, BtTraits>::~block_map() {

    do_clear();
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::create(const index<N> &idx) {

    static const char *method = "create(const index<N>&)";

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    block_type *ptr = m_bf.create_block(idx);
    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    typename map_type::iterator i = m_map.find(aidx);
    if(i == m_map.end()) {
        m_map.insert(std::make_pair(aidx, ptr));
    } else {
        m_bf.destroy_block(i->second);
        i->second = ptr;
    }
    m_dirty_cache = true;
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::remove(const index<N> &idx) {

    static const char *method = "remove(const index<N>&)";

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    typename map_type::iterator i = m_map.find(aidx);
    if(i != m_map.end()) {
        m_bf.destroy_block(i->second);
        m_map.erase(i);
    }
    m_dirty_cache = true;
}


template<size_t N, typename BtTraits>
bool block_map<N, BtTraits>::contains(const index<N> &idx) const {

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    return m_map.find(aidx) != m_map.end();
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::get_all(std::vector<size_t> &blst) const {

    if(m_dirty_cache) {
        m_cached_blst.clear();
        m_cached_blst.reserve(m_map.size());
        for(typename map_type::const_iterator i = m_map.begin();
            i != m_map.end(); ++i) {
            m_cached_blst.push_back(i->first);
        }
    }
    blst = m_cached_blst;
}


template<size_t N, typename BtTraits>
typename block_map<N, BtTraits>::block_type&
block_map<N, BtTraits>::get(const index<N> &idx) {

    static const char *method = "get(const index<N>&)";

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    typename map_type::iterator i = m_map.find(aidx);
    if(i == m_map.end()) {
        throw block_not_found(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Requested block cannot be located.");
    }

    return *(i->second);
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::clear() {

    static const char *method = "clear()";

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    do_clear();
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::on_set_immutable() {

    for(typename map_type::iterator i = m_map.begin(); i != m_map.end(); ++i) {
        i->second->set_immutable();
    }
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::do_clear() {

    for(typename map_type::iterator i = m_map.begin(); i != m_map.end(); ++i) {
        m_bf.destroy_block(i->second);
        i->second = 0;
    }
    m_map.clear();
    m_dirty_cache = true;
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_IMPL_H
