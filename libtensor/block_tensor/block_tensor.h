#ifndef LIBTENSOR_BLOCK_TENSOR_H
#define LIBTENSOR_BLOCK_TENSOR_H

#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space.h>
#include <libtensor/core/immutable.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/mp/default_sync_policy.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "../gen_block_tensor/auto_rwlock.h"
#include "../gen_block_tensor/impl/block_map_impl.h"
#include "block_tensor_traits.h"

namespace libtensor {

/** \brief Block %tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    \ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc>
class block_tensor : public block_tensor_i<N, T>, public immutable {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef block_tensor_traits<T, Alloc> bt_traits;

public:
    block_index_space<N> m_bis; //!< Block %index space
    dimensions<N> m_bidims; //!< Block %index %dimensions
    symmetry<N, T> m_symmetry; //!< Block %tensor symmetry
    orbit_list<N, T> *m_orblst; //!< Orbit list
    block_map<N, T, bt_traits> m_map; //!< Block map
    libutil::rwlock m_lock; //!< Read-write lock

public:
    //!    \name Construction and destruction
    //@{
    block_tensor(const block_index_space<N> &bis);
    block_tensor(const block_tensor<N, T, Alloc> &bt);
    virtual ~block_tensor();
    //@}

    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{
    virtual const block_index_space<N> &get_bis() const;
    //@}

protected:
    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{
    virtual const symmetry<N, T> &on_req_const_symmetry();
    virtual symmetry<N, T> &on_req_symmetry();
    virtual dense_tensor_i<N, T> &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    virtual void on_req_sync_on();
    virtual void on_req_sync_off();
    //@}

    //!    \name Implementation of libtensor::immutable
    //@{
    virtual void on_set_immutable();
    //@}

private:
    void update_orblst(auto_rwlock &lock);
};


template<size_t N, typename T, typename Alloc>
const char *block_tensor<N, T, Alloc>::k_clazz = "block_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_index_space<N> &bis) :
    m_bis(bis),
    m_bidims(bis.get_block_index_dims()),
    m_symmetry(m_bis),
    m_orblst(0),
    m_map(m_bis) {

}


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::block_tensor(const block_tensor<N, T, Alloc> &bt) :

    m_bis(bt.get_bis()),
    m_bidims(bt.m_bidims),
    m_symmetry(bt.get_bis()),
    m_orblst(0),
    m_map(m_bis) {

}


template<size_t N, typename T, typename Alloc>
block_tensor<N, T, Alloc>::~block_tensor() {

    delete m_orblst;
}


template<size_t N, typename T, typename Alloc>
const block_index_space<N> &block_tensor<N, T, Alloc>::get_bis()
    const {

    return m_bis;
}


template<size_t N, typename T, typename Alloc>
const symmetry<N, T> &block_tensor<N, T, Alloc>::on_req_const_symmetry() {

    return m_symmetry;
}


template<size_t N, typename T, typename Alloc>
symmetry<N, T> &block_tensor<N, T, Alloc>::on_req_symmetry() {

    static const char *method = "on_req_symmetry()";

    auto_rwlock lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "symmetry");
    }

    lock.upgrade();
    delete m_orblst;
    m_orblst = 0;

    return m_symmetry;
}


template<size_t N, typename T, typename Alloc>
dense_tensor_i<N, T> &block_tensor<N, T, Alloc>::on_req_const_block(
    const index<N> &idx) {

    return on_req_block(idx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_ret_const_block(const index<N> &idx) {

    on_ret_block(idx);
}


template<size_t N, typename T, typename Alloc>
dense_tensor_i<N, T> &block_tensor<N, T, Alloc>::on_req_block(
    const index<N> &idx) {

    static const char *method = "on_req_block(const index<N>&)";

    auto_rwlock lock(m_lock);

    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    if(!m_map.contains(idx)) {
        lock.upgrade();
        m_map.create(idx);
    }
    return m_map.get(idx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx) {

}


template<size_t N, typename T, typename Alloc>
bool block_tensor<N, T, Alloc>::on_req_is_zero_block(
    const index<N> &idx) {

    static const char *method = "on_req_is_zero_block(const index<N>&)";

    auto_rwlock lock(m_lock);

    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method,
            __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    return !m_map.contains(idx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_zero_block(const index<N> &idx) {

    static const char *method = "on_req_zero_block(const index<N>&)";

    auto_rwlock lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    update_orblst(lock);
    if(!m_orblst->contains(idx)) {
        throw symmetry_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Index does not correspond to a canonical block.");
    }
    lock.upgrade();
    m_map.remove(idx);
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_zero_all_blocks() {

    static const char *method = "on_req_zero_all_blocks()";

    auto_rwlock lock(m_lock);

    if(is_immutable()) {
        throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Immutable object cannot be modified.");
    }
    lock.upgrade();
    m_map.clear();
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_sync_on() {

}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::on_req_sync_off() {

}


template<size_t N, typename T, typename Alloc>
inline void block_tensor<N, T, Alloc>::on_set_immutable() {

    auto_rwlock lock(m_lock);

    lock.upgrade();
    m_map.set_immutable();
}


template<size_t N, typename T, typename Alloc>
void block_tensor<N, T, Alloc>::update_orblst(auto_rwlock &lock) {

    if(m_orblst == 0) {
        lock.upgrade();
        //  We may have waited long enough here that the orbit has been updated
        //  in another thread already. Need to double check.
        if(m_orblst == 0) m_orblst = new orbit_list<N, T>(m_symmetry);
        lock.downgrade();
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BLOCK_TENSOR_H
