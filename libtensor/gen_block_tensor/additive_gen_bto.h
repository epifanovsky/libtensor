#ifndef LIBTENSOR_ADDITIVE_GEN_BTO_H
#define LIBTENSOR_ADDITIVE_GEN_BTO_H

#include <libtensor/core/tensor_transf.h>
#include "direct_gen_bto.h"

namespace libtensor {


/** \brief Base class for additive block %tensor operations
    \tparam N Tensor order.

    Additive block %tensor operations are those that can add their result
    to the output block %tensor as opposed to simply replacing it. This
    class extends direct_bto<N, Traits> with two new functions: one is invoked
    to perform the block %tensor operation additively, the other does that
    for only one canonical block.

    The scalar transformation provided in both functions transforms the
    elements of the result of the operation before adding it to the output
    block %tensor.

    \ingroup libtensor_btod
 **/
template<size_t N, typename BtiTraits>
class additive_gen_bto: public direct_gen_bto<N, BtiTraits> {
public:
    //! Type of tensor elements
    typedef typename BtiTraits::element_type element_type;

    //! Type of blocks of block tensors
    typedef typename BtiTraits::template wr_block_type<N>::type wr_block_type;

    //! Type of tensor transformation
    typedef tensor_transf<N, element_type> tensor_transf_type;

public:
    virtual void perform(gen_block_stream_i<N, BtiTraits> &out) = 0;

    /** \brief Compute the result and store it in the output block tensor
        \param bt Output block tensor.
     **/
    virtual void perform(gen_block_tensor_i<N, BtiTraits> &bt) = 0;

    /** \brief Computes the result of the operation and adds it to the
            output block tensor
        \param bt Output block tensor.
        \param c Scalar transformation of result.
     **/
    virtual void perform(gen_block_tensor_i<N, BtiTraits> &bt,
            const scalar_transf<element_type> &c) = 0;

    /** \brief Computes a single block of the result and adds it to
            the output %tensor
        \param zero Zero out the output before the computation.
        \param idx Index of the block to compute.
        \param tr Transformation of the block.
        \param blk Output %tensor.
     **/
    virtual void compute_block(
            bool zero,
            const index<N> &idx,
            const tensor_transf_type &tr,
            wr_block_type &blk) = 0;

    /** \brief Implementation of direct_gen_bto<N, BtiTraits>::compute_block
     **/
    virtual void compute_block(
            const index<N> &idx,
            wr_block_type &blk) {

        compute_block(true, idx, tensor_transf_type(), blk);
    }

};


} // namespace libtensor


#endif // LIBTENSOR_ADDITIVE_BTO_H
