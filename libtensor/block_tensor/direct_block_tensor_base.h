#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_BASE_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_BASE_H

#include <libtensor/gen_block_tensor/direct_gen_bto.h>
#include <libtensor/block_tensor/btod/btod_traits.h>

namespace libtensor {

/** \brief Direct block %tensor base
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator type.

    Base class for direct block %tensors. Implements the functions

    \ingroup libtensor_core
**/
template<size_t N, typename T>
class direct_block_tensor_base : public block_tensor_i<N, T> {
public:
    static const char *k_clazz; //!< Class name

public:
    typedef T element_t; //!< Tensor element type
    typedef direct_gen_bto<N, btod_traits::bti_traits> operation_t;

protected:
    //!    Underlying block tensor operation
    operation_t &m_op;

public:
    //!    \name Construction and destruction
    //@{

    direct_block_tensor_base(operation_t &op) :
        m_op(op) { }

    virtual ~direct_block_tensor_base() { }

    //@}

    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{

    virtual const block_index_space<N> &get_bis() const;

    //@}

protected:
    operation_t &get_op() const {
        return m_op;
    }

protected:
    //!    \name Implementation of libtensor::block_tensor_i<N, T>
    //@{

    virtual symmetry<N, T> &on_req_symmetry();
    virtual const symmetry<N, T> &on_req_const_symmetry();

    virtual void on_req_zero_block(const index<N> &idx);
    virtual void on_req_zero_all_blocks();

    virtual bool on_req_is_zero_block(const index<N> &idx) = 0;
    virtual dense_tensor_i<N, T> &on_req_const_block(const index<N> &idx) = 0;
    virtual void on_ret_const_block(const index<N> &idx) = 0;
    virtual dense_tensor_i<N, T> &on_req_block(const index<N> &idx) = 0;
    virtual void on_ret_block(const index<N> &idx) = 0;

    //@}
};


template<size_t N, typename T>
const char *direct_block_tensor_base<N, T>::k_clazz =
    "direct_block_tensor_base<N, T>";


template<size_t N, typename T>
const block_index_space<N> &direct_block_tensor_base<N, T>::get_bis() const {

    return m_op.get_bis();
}


template<size_t N, typename T>
symmetry<N, T> &direct_block_tensor_base<N, T>::on_req_symmetry() {

    static const char *method = "on_req_const_symmetry()";

    throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
        "direct_block_tensor");
}


template<size_t N, typename T>
const symmetry<N, T> &direct_block_tensor_base<N, T>::on_req_const_symmetry() {

    return m_op.get_symmetry();
}


template<size_t N, typename T>
void direct_block_tensor_base<N, T>::on_req_zero_block(const index<N> &idx) {

    static const char *method = "on_req_zero_block(const index<N>&)";

    throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
        "direct_block_tensor");
}


template<size_t N, typename T>
void direct_block_tensor_base<N, T>::on_req_zero_all_blocks() {

    static const char *method = "on_req_zero_all_blocks()";

    throw immut_violation(g_ns, k_clazz, method, __FILE__, __LINE__,
        "direct_block_tensor");
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_BASE_H
