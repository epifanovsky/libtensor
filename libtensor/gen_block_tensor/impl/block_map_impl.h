#ifndef LIBTENSOR_BLOCK_MAP_IMPL_H
#define LIBTENSOR_BLOCK_MAP_IMPL_H

#include <algorithm>
#include <libtensor/core/abs_index.h>
#include "../block_map.h"

namespace libtensor {


template<size_t N, typename BtTraits>
const char block_map<N, BtTraits>::k_clazz[] = "block_map<N, BtTraits>";


template<size_t N, typename BtTraits>
block_map<N, BtTraits>::~block_map() {

    do_clear();
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::create(const index<N> &idx) {

    static const char method[] = "create(const index<N>&)";

    typedef typename std::vector<pair_type>::iterator iterator_type;

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    block_type *ptr = m_bf.create_block(idx);

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    iterator_type i = std::lower_bound(m_blocks.begin(), m_blocks.end(),
        pair_type(aidx, 0), blkmap_cmp());
    if(i != m_blocks.end() && i->first == aidx) {
        m_bf.destroy_block(i->second);
        i->second = ptr;
    } else {
        bool sorted = m_sorted;
        if(m_blocks.size() > 0) {
            if(sorted && aidx < m_blocks.back().first) sorted = false;
        }
        m_blocks.push_back(pair_type(aidx, ptr));
        m_sorted = sorted;
    }
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::remove(const index<N> &idx) {

    static const char method[] = "remove(const index<N>&)";

    typedef typename std::vector<pair_type>::iterator iterator_type;

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    if(!m_sorted) sort();

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    iterator_type i = std::lower_bound(m_blocks.begin(), m_blocks.end(),
        pair_type(aidx, 0), blkmap_cmp());
    if(i != m_blocks.end() && i->first == aidx) {
        m_bf.destroy_block(i->second);
        m_blocks.erase(i);
    }
}


template<size_t N, typename BtTraits>
bool block_map<N, BtTraits>::contains(const index<N> &idx) const {

    if(!m_sorted) sort();

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    return std::binary_search(m_blocks.begin(), m_blocks.end(),
        pair_type(aidx, 0), blkmap_cmp());
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::get_all(std::vector<size_t> &blst) const {

    typedef typename std::vector<pair_type>::iterator iterator_type;

    if(!m_sorted) sort();

    blst.clear();
    blst.reserve(m_blocks.size());
    for(iterator_type i = m_blocks.begin(); i != m_blocks.end(); ++i) {
        blst.push_back(i->first);
    }
}


template<size_t N, typename BtTraits>
typename block_map<N, BtTraits>::block_type&
block_map<N, BtTraits>::get(const index<N> &idx) {

    static const char method[] = "get(const index<N>&)";

    typedef typename std::vector<pair_type>::iterator iterator_type;

    if(!m_sorted) sort();

    size_t aidx = abs_index<N>::get_abs_index(idx, m_bidims);
    iterator_type i = std::lower_bound(m_blocks.begin(), m_blocks.end(),
        pair_type(aidx, 0), blkmap_cmp());
    if(i == m_blocks.end() || i->first != aidx) {
        throw block_not_found(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Requested block cannot be located.");
    }

    return *(i->second);
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::clear() {

    static const char method[] = "clear()";

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "this");
    }

    do_clear();
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::on_set_immutable() {

    typedef typename std::vector<pair_type>::iterator iterator_type;

    for(iterator_type i = m_blocks.begin(); i != m_blocks.end(); ++i) {
        i->second->set_immutable();
    }
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::sort() const {

    std::sort(m_blocks.begin(), m_blocks.end(), blkmap_cmp());
    m_sorted = true;
}


template<size_t N, typename BtTraits>
void block_map<N, BtTraits>::do_clear() {

    typedef typename std::vector<pair_type>::iterator iterator_type;

    for(iterator_type i = m_blocks.begin(); i != m_blocks.end(); ++i) {
        m_bf.destroy_block(i->second);
        i->second = 0;
    }
    m_blocks.clear();
    m_sorted = true;
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_MAP_IMPL_H
