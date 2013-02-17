#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H

#include <libutil/threads/auto_lock.h>
#include <libtensor/core/short_orbit.h>
#include "../gen_block_tensor.h"

namespace libtensor {


template<size_t N, typename BtTraits>
const char gen_block_tensor<N, BtTraits>::k_clazz[] =
    "gen_block_tensor<N, BtTraits>";


template<size_t N, typename BtTraits>
gen_block_tensor<N, BtTraits>::gen_block_tensor(
    const block_index_space<N> &bis) :

    m_bis(bis),
    m_bidims(bis.get_block_index_dims()),
    m_symmetry(m_bis),
    m_map(m_bis) {

}


template<size_t N, typename BtTraits>
gen_block_tensor<N, BtTraits>::~gen_block_tensor() {

}


template<size_t N, typename BtTraits>
const block_index_space<N> &gen_block_tensor<N, BtTraits>::get_bis() const {

    return m_bis;
}


template<size_t N, typename BtTraits>
const typename gen_block_tensor<N, BtTraits>::symmetry_type&
gen_block_tensor<N, BtTraits>::on_req_const_symmetry() {

    return m_symmetry;
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::symmetry_type&
gen_block_tensor<N, BtTraits>::on_req_symmetry() {

    static const char method[] = "on_req_symmetry()";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "symmetry");
    }

    return m_symmetry;
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::rd_block_type&
gen_block_tensor<N, BtTraits>::on_req_const_block(const index<N> &idx) {

    return get_block(idx, false);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_ret_const_block(const index<N> &idx) {

    on_ret_block(idx);
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::wr_block_type&
gen_block_tensor<N, BtTraits>::on_req_block(const index<N> &idx) {

    return get_block(idx, true);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_ret_block(const index<N> &idx) {

}


template<size_t N, typename BtTraits>
bool gen_block_tensor<N, BtTraits>::on_req_is_zero_block(const index<N> &idx) {

    static const char method[] = "on_req_is_zero_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(!check_canonical_block(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }

    return !m_map.contains(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_req_nonzero_blocks(
    std::vector<size_t> &nzlst) {

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    m_map.get_all(nzlst);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_req_zero_block(const index<N> &idx) {

    static const char method[] = "on_req_zero_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }

    if(!check_canonical_block(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }

    m_map.remove(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_req_zero_all_blocks() {

    static const char method[] = "on_req_zero_all_blocks()";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    m_map.clear();
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_set_immutable() {

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    m_map.set_immutable();
}


template<size_t N, typename BtTraits>
bool gen_block_tensor<N, BtTraits>::check_canonical_block(
    const index<N> &idx) {

#ifndef LIBTENSOR_DEBUG
    //  This check is fast, but not very robust. Disable it in the debug mode.
    if(m_map.contains(idx)) return true;
#endif // LIBTENSOR_DEBUG

    short_orbit<N, element_type> o(m_symmetry, idx, true);
    return o.is_allowed() && o.get_cindex().equals(idx);
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::block_type&
gen_block_tensor<N, BtTraits>::get_block(const index<N> &idx, bool create) {

    static const char method[] = "get_block(const index<N>&, bool)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(!check_canonical_block(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }

    if(!m_map.contains(idx)) {
        if(create) {
            m_map.create(idx);
        } else {
            throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Block does not exist.");
        }
    }
    return m_map.get(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H
