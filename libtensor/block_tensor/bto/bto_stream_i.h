#ifndef LIBTENSOR_BTO_STREAM_I_H
#define LIBTENSOR_BTO_STREAM_I_H

#include <libtensor/core/index.h>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


/** \brief Interface for block tensor operations that accept a stream of blocks
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits structure.

    Through this interface, a block tensor operation accepts the blocks of
    a block tensor from an external source. Likewise, block tensor operations
    can act as the source of blocks an put() them for further processing.

    Block tensor operations that implement this interface can be chained
    to act as a series of filters.

    Methods open() and close() shall be called with external synchronization
    and therefore need not be thread-safe.

    The implementation of put() must be thread-safe. Multiple blocks with
    the same value of index may arrive, in which case the implementation
    shall process them according to its own policy.

    \sa bto_traits

    \ingroup libtensor_block_tensor_bto
 **/
template<size_t N, typename Traits>
class bto_stream_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~bto_stream_i() { }

    /** \brief Opens the stream
     **/
    virtual void open() = 0;

    /** \brief Closes the stream
     **/
    virtual void close() = 0;

    /** \brief Puts a block into the stream
        \param idx Index of the block.
        \param blk Block.
        \param tr Transformation of the block.
     **/
    virtual void put(
        const index<N> &idx,
        block_type &blk,
        const tensor_transf_type &tr) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_STREAM_I_H
