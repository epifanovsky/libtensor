#ifndef LIBTENSOR_DIAG_BLOCK_TENSOR_H
#define LIBTENSOR_DIAG_BLOCK_TENSOR_H

#include <libtensor/core/immutable.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/gen_block_tensor.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "diag_block_tensor_i.h"
#include "diag_block_tensor_traits.h"

namespace libtensor {


/** \brief Diagonal block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator.

    \ingroup libtensor_diag_block_tensor
 **/
template<size_t N, typename T, typename Alloc>
class diag_block_tensor :
    public diag_block_tensor_i<N, T>,
    public immutable,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef diag_block_tensor_traits<T, Alloc> bt_traits;
    typedef typename bt_traits::bti_traits bti_traits;

public:
    gen_block_tensor<N, bt_traits> m_bt;
    gen_block_tensor_ctrl<N, bti_traits> m_ctrl;

public:
    /** \brief Initializes this diagonal block tensor in a tensor space
     **/
    diag_block_tensor(const block_index_space<N> &bis);

    /** \brief Initializes this diagonal block tensor in the space of another
            tensor (data is not copied)
     **/
    diag_block_tensor(const diag_block_tensor<N, T, Alloc> &bt);

    /** \brief Destroys this tensor
     **/
    virtual ~diag_block_tensor();

    /** \brief Returns the block index space of this tensor
     **/
    virtual const block_index_space<N> &get_bis() const;

protected:
    virtual const symmetry<N, T> &on_req_const_symmetry();
    virtual symmetry<N, T> &on_req_symmetry();
    virtual diag_tensor_i<N, T> &on_req_const_block(const index<N> &idx);
    virtual void on_ret_const_block(const index<N> &idx);
    virtual diag_tensor_i<N, T> &on_req_block(const index<N> &idx);
    virtual void on_ret_block(const index<N> &idx);
    virtual bool on_req_is_zero_block(const index<N> &idx);
    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();
    virtual void on_set_immutable();

private:
    void update_orblst(auto_rwlock &lock);

};


template<size_t N, typename T, typename Alloc>
const char *diag_block_tensor<N, T, Alloc>::k_clazz =
    "diag_block_tensor<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
diag_block_tensor<N, T, Alloc>::diag_block_tensor(
    const block_index_space<N> &bis) :

    m_bt(bis), m_ctrl(m_bt) {

}


template<size_t N, typename T, typename Alloc>
diag_block_tensor<N, T, Alloc>::diag_block_tensor(
    const diag_block_tensor<N, T, Alloc> &bt) :

    m_bt(bt.get_bis()), m_ctrl(m_bt) {

}


template<size_t N, typename T, typename Alloc>
diag_block_tensor<N, T, Alloc>::~diag_block_tensor() {

}


template<size_t N, typename T, typename Alloc>
const block_index_space<N> &diag_block_tensor<N, T, Alloc>::get_bis() const {

    return m_bt.get_bis();
}


template<size_t N, typename T, typename Alloc>
const symmetry<N, T> &diag_block_tensor<N, T, Alloc>::on_req_const_symmetry() {

    return m_ctrl.req_const_symmetry();
}


template<size_t N, typename T, typename Alloc>
symmetry<N, T> &diag_block_tensor<N, T, Alloc>::on_req_symmetry() {

    return m_ctrl.req_symmetry();
}


template<size_t N, typename T, typename Alloc>
diag_tensor_i<N, T> &diag_block_tensor<N, T, Alloc>::on_req_const_block(
    const index<N> &idx) {

    return m_ctrl.req_const_block(idx);
}


template<size_t N, typename T, typename Alloc>
void diag_block_tensor<N, T, Alloc>::on_ret_const_block(const index<N> &idx) {

    m_ctrl.ret_const_block(idx);
}


template<size_t N, typename T, typename Alloc>
diag_tensor_i<N, T> &diag_block_tensor<N, T, Alloc>::on_req_block(
    const index<N> &idx) {

    return m_ctrl.req_block(idx);
}


template<size_t N, typename T, typename Alloc>
void diag_block_tensor<N, T, Alloc>::on_ret_block(const index<N> &idx) {

    m_ctrl.ret_block(idx);
}


template<size_t N, typename T, typename Alloc>
bool diag_block_tensor<N, T, Alloc>::on_req_is_zero_block(
    const index<N> &idx) {

    return m_ctrl.req_is_zero_block(idx);
}


template<size_t N, typename T, typename Alloc>
void diag_block_tensor<N, T, Alloc>::on_req_zero_block(const index<N> &idx) {

    m_ctrl.req_zero_block(idx);
}


template<size_t N, typename T, typename Alloc>
void diag_block_tensor<N, T, Alloc>::on_req_zero_all_blocks() {

    m_ctrl.req_zero_all_blocks();
}


template<size_t N, typename T, typename Alloc>
void diag_block_tensor<N, T, Alloc>::on_set_immutable() {

    m_bt.set_immutable();
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BLOCK_TENSOR_H
