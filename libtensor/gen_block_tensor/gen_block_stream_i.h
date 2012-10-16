#ifndef LIBTENSOR_GEN_BLOCK_STREAM_I_H
#define LIBTENSOR_GEN_BLOCK_STREAM_I_H

#include <libtensor/core/index.h>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


/** \brief Output stream of block tensor blocks
    \tparam N Tensor order.
    \tparam BtiTraits Block tensor interface traits.

    This interface allows the user to push computed blocks for further
    processing. All kinds of data sources, including block tensor operations,
    can be the users of this interface.

    The user shall invoke open() before putting any blocks into the stream,
    and call close() after finishing the work. The user shall call put()
    to push blocks. put() may be called multiple times with the same index,
    and not all possible index values may be covered in a session.

    Certain block tensor operations may implement this interface and be chained
    to act as a series of filters.

    <b>Thread safety</b>

    Methods open() and close() shall be called with external synchronization
    and therefore need not be thread-safe.

    The implementation of put() must be thread-safe. Multiple blocks with
    the same value of index may arrive, in which case the implementation
    shall process them according to its own policy.

    \sa gen_block_tensor_i

    \ingroup libtensor_gen_block_tensor
 **/
template<size_t N, typename BtiTraits>
class gen_block_stream_i {
public:
    typedef typename BtiTraits::element_type element_type;
    typedef typename BtiTraits::template rd_block_type<N>::type rd_block_type;

public:
    /** \brief Virtual destructor
     **/
    virtual ~gen_block_stream_i() { }

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
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr) = 0;

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BLOCK_STREAM_I_H
