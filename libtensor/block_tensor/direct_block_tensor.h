#ifndef LIBTENSOR_DIRECT_BLOCK_TENSOR_H
#define LIBTENSOR_DIRECT_BLOCK_TENSOR_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/gen_block_tensor/direct_gen_block_tensor.h>
#include <libtensor/gen_block_tensor/impl/direct_gen_block_tensor_impl.h>
#include <libtensor/gen_block_tensor/direct_gen_bto.h>
#include "block_tensor_traits.h"

namespace libtensor {


/** \brief Direct block tensor
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Alloc Memory allocator type.

    \ingroup libtensor_core
 **/
template<size_t N, typename T, typename Alloc>
class direct_block_tensor :
    public block_tensor_rd_i<N, T>,
    public noncopyable {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef block_tensor_traits<T, Alloc> bt_traits;
    typedef typename bt_traits::bti_traits bti_traits;
    typedef direct_gen_bto<N, bti_traits> operation_t;

private:
    direct_gen_block_tensor<N, bt_traits> m_gbt;
    gen_block_tensor_rd_ctrl<N, bti_traits> m_ctrl;

public:
    //!    \name Construction and destruction
    //@{

    direct_block_tensor(operation_t &op) : m_gbt(op), m_ctrl(m_gbt) { }

    virtual ~direct_block_tensor() { }
    //@}

    virtual const block_index_space<N> &get_bis() const {
        return m_gbt.get_bis();
    }

protected:
    //!    \name Implementation of libtensor::block_tensor_rd_i<N, T>
    //@{

    virtual const symmetry<N, double> &on_req_const_symmetry() {
        return m_ctrl.req_const_symmetry();
    }

    virtual bool on_req_is_zero_block(const index<N> &idx) {
        return m_ctrl.req_is_zero_block(idx);
    }

    virtual dense_tensor_rd_i<N, T> &on_req_const_block(const index<N> &idx) {
        return m_ctrl.req_const_block(idx);
    }

    virtual void on_ret_const_block(const index<N> &idx) {
        m_ctrl.ret_const_block(idx);
    }

    //@}

    operation_t &get_op() {
        return m_gbt.get_op();
    }
};


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_H
