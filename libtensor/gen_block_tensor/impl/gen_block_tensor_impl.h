#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H

#include "../gen_block_tensor.h"

namespace libtensor {


template<size_t N, typename BtTraits>
const char *gen_block_tensor<N, BtTraits>::k_clazz =
    "gen_block_tensor<N, BtTraits>";


template<size_t N, typename BtTraits>
gen_block_tensor<N, BtTraits>::gen_block_tensor(
    const block_index_space<N> &bis) :

    m_bis(bis),
    m_bidims(bis.get_block_index_dims()),
    m_symmetry(m_bis),
    m_orblst(0),
    m_map(m_bis) {

}


template<size_t N, typename BtTraits>
gen_block_tensor<N, BtTraits>::~gen_block_tensor() {

    delete m_orblst; m_orblst = 0;
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

    static const char *method = "on_req_symmetry()";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "symmetry");
    }

    delete m_orblst;
    m_orblst = 0;

    return m_symmetry;
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::rd_block_type&
gen_block_tensor<N, BtTraits>::on_req_const_block(const index<N> &idx) {

    return get_block(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_ret_const_block(const index<N> &idx) {

    on_ret_block(idx);
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::wr_block_type&
gen_block_tensor<N, BtTraits>::on_req_block(const index<N> &idx) {

    return get_block(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_ret_block(const index<N> &idx) {

}


template<size_t N, typename BtTraits>
bool gen_block_tensor<N, BtTraits>::on_req_is_zero_block(const index<N> &idx) {

    static const char *method = "on_req_is_zero_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    update_orblst();
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    return !m_map.contains(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_req_zero_block(const index<N> &idx) {

    static const char *method = "on_req_zero_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    update_orblst();
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    m_map.remove(idx);
}


template<size_t N, typename BtTraits>
void gen_block_tensor<N, BtTraits>::on_req_zero_all_blocks() {

    static const char *method = "on_req_zero_all_blocks()";

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
void gen_block_tensor<N, BtTraits>::update_orblst() {

    if(m_orblst == 0) {
        //  We may have waited long enough here that the orbit has been updated
        //  in another thread already. Need to double check.
        if(m_orblst == 0) {
            m_orblst = new orbit_list<N, element_type>(m_symmetry);
        }
    }
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::block_type&
gen_block_tensor<N, BtTraits>::get_block(const index<N> &idx) {

    static const char *method = "get_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    update_orblst();
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }

    if(!m_map.contains(idx)) {
        if(!m_map.contains(idx)) m_map.create(idx);
    }
    return m_map.get(idx);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H
