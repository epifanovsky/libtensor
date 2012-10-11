#ifndef LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_BASE_H
#define LIBTENSOR_DIRECT_GEN_BLOCK_TENSOR_BASE_H

#include "direct_gen_bto.h"
#include "gen_block_tensor_i.h"


namespace libtensor {

/** \brief Direct generalized block %tensor base
    \tparam N Tensor order.
    \tparam BtiTraits Block %tensor interface traits.

    Base class for direct block %tensors. Implements the functions

    \ingroup libtensor_core
**/
template<size_t N, typename BtiTraits>
class direct_gen_block_tensor_base :
    public gen_block_tensor_rd_i<N, BtiTraits> {
public:
    static const char *k_clazz; //!< Class name

public:
    //! Tensor element type
    typedef typename BtiTraits::element_type element_type;

    //! Type of read-only blocks
    typedef typename BtiTraits::template rd_block_type<N>::type rd_block_type;

    //! Block tensor operation
    typedef direct_gen_bto<N, BtiTraits> operation_t;

protected:
    //!    Underlying block tensor operation
    operation_t &m_op;

public:
    //!    \name Construction and destruction
    //@{

    direct_gen_block_tensor_base(operation_t &op) : m_op(op) { }

    virtual ~direct_gen_block_tensor_base() { }

    //@}

    //! \name Implementation of libtensor::gen_block_tensor_rd_i<N, BtiTraits>
    //@{

    virtual const block_index_space<N> &get_bis() const;

    //@}

protected:
    operation_t &get_op() const {
        return m_op;
    }

protected:
    //! \name Implementation of libtensor::gen_block_tensor_rd_i<N, BtiTraits>
    //@{

    virtual const symmetry<N, element_type> &on_req_const_symmetry();

    virtual bool on_req_is_zero_block(const index<N> &idx) = 0;

    virtual rd_block_type &on_req_const_block(const index<N> &idx) = 0;

    virtual void on_ret_const_block(const index<N> &idx) = 0;

    //@}
};


template<size_t N, typename BtiTraits>
const char *direct_gen_block_tensor_base<N, BtiTraits>::k_clazz =
    "direct_gen_block_tensor_base<N, BtiTraits>";


template<size_t N, typename BtiTraits>
const block_index_space<N> &
direct_gen_block_tensor_base<N, BtiTraits>::get_bis() const {

    return m_op.get_bis();
}


template<size_t N, typename BtiTraits>
const symmetry<N, typename BtiTraits::element_type> &
direct_gen_block_tensor_base<N, BtiTraits>::on_req_const_symmetry() {

    return m_op.get_symmetry();
}


} // namespace libtensor

#endif // LIBTENSOR_DIRECT_BLOCK_TENSOR_BASE_H
