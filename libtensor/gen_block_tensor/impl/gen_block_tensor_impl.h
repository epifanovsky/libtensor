#ifndef LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H
#define LIBTENSOR_GEN_BLOCK_TENSOR_IMPL_H

#include <algorithm> // for std::swap
#include <libutil/threads/auto_lock.h>
#include <libtensor/core/short_orbit.h>
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
    m_orblst(0), m_orblst_inprogress(false),
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

    static const char *method = "on_req_zero_block(const index<N>&)";

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
bool gen_block_tensor<N, BtTraits>::update_orblst(bool wait) {

    //  If wait = true, return only when new orblst is built,
    //  always return true.
    //  If wait = false, do not wait for orblst to be built. Return true if
    //  it is ready, false otherwise.

    //  m_lock is locked by the caller

    if(m_orblst) return true;
    if(!wait && m_orblst_inprogress) return false;

    //  If building of orblst is in progress in another thread, this thread
    //  still builds its own. It may occur that multiple threads are building
    //  orblst, which adds overhead. That however greatly simplifies
    //  synchronization.

    m_orblst_inprogress = true;
    m_lock.unlock();
    orbit_list<N, element_type> *orblst = 0;
    try {
        orblst = new orbit_list<N, element_type>(m_symmetry);
    } catch(...) {
        //  Make sure this function returns with m_lock locked
        m_lock.lock();
        throw;
    }
    m_lock.lock();
    if(m_orblst_inprogress && m_orblst == 0) {
        //  This thread won the race!
        std::swap(orblst, m_orblst);
        m_orblst_inprogress = false;
    }
    delete orblst;

    return true;
}


template<size_t N, typename BtTraits>
bool gen_block_tensor<N, BtTraits>::check_canonical_block(const index<N> &idx) {

    //  This trick here helps to reduce lock contention when threads
    //  need to wait for orblst to be built. Getting rid of this symmetry
    //  violation check entirely would help, but we want to stay strict.

    bool use_orblst = update_orblst(false);
    if(use_orblst) {
        if(!m_orblst->contains(idx)) return false;
    } else {
        short_orbit<N, element_type> o(m_symmetry, idx, true);
        if(!o.is_allowed() || !o.get_cindex().equals(idx)) return false;
    }
    return true;
}


template<size_t N, typename BtTraits>
typename gen_block_tensor<N, BtTraits>::block_type&
gen_block_tensor<N, BtTraits>::get_block(const index<N> &idx) {

    static const char *method = "get_block(const index<N>&)";

    libutil::auto_lock<libutil::mutex> lock(m_lock);

    if(!check_canonical_block(idx)) {
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
