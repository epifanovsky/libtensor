#ifndef LIBTENSOR_ADDITIVE_BTO_H
#define LIBTENSOR_ADDITIVE_BTO_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/gen_block_tensor/direct_gen_bto.h>

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
template<size_t N, typename Traits>
class additive_bto: public direct_gen_bto<N, typename Traits::bti_traits> {
public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_t;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of block tensors
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_t;

    //! Type of blocks of block tensors
    typedef typename Traits::template block_type<N>::type block_t;

public:
    virtual void perform(gen_block_stream_i<N, bti_traits> &out) = 0;
    virtual void perform(block_tensor_t &bt) = 0;

    /** \brief Computes the result of the operation and adds it to the
            output block tensor
        \param bt Output block tensor.
        \param c Scaling coefficient.
     **/
    virtual void perform(block_tensor_t &bt, const element_t &c) = 0;

    /** \brief Computes a single block of the result and adds it to
            the output %tensor
        \param zero Zero out the output before the computation.
        \param blk Output %tensor.
        \param i Index of the block to compute.
        \param tr Transformation of the block.
        \param c Scaling coefficient.
     **/
    virtual void compute_block(bool zero, block_t &blk, const index<N> &i,
        const tensor_transf<N, element_t> &tr, const element_t &c) = 0;
    virtual void compute_block(block_t &blk, const index<N> &idx) {
        compute_block(true, blk, idx, tensor_transf<N, element_t>(),
            Traits::identity());
    }

protected:
    /** \brief Invokes compute_block on another additive operation;
            allows derived classes to call other additive operations
     **/
    void compute_block(additive_bto<N, Traits> &op, bool zero, block_t &blk,
        const index<N> &i, const tensor_transf<N, element_t> &tr,
        const element_t &c);

};


} // namespace libtensor


#endif // LIBTENSOR_ADDITIVE_BTO_H
