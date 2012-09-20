#ifndef LIBTENSOR_BTO_AUX_COPY_H
#define LIBTENSOR_BTO_AUX_COPY_H

#include "bto_stream_i.h"

namespace libtensor {


/** \brief Saves blocks into a block tensor (auxiliary operation)
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits structure.

    This auxiliary block tensor operation accepts blocks and saves them into
    a given block tensor object. Upon initialization via open(), the operation
    zeroes out the target block tensor and installs the specified symmetry.
    Each put() will copy the block with the given index. If put() is called
    multiple times with the same block index, only the last one will be saved.
    The blocks must be canonical.

    \sa bto_stream_i

    \ingroup libtensor_block_tensor_bto
 **/
template<size_t N, typename Traits>
class bto_aux_copy : public bto_stream_i<N, Traits> {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::template block_type<N>::type block_type;
    typedef typename Traits::template block_tensor_type<N>::type
        block_tensor_type;
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_type;
    typedef symmetry<N, element_type> symmetry_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

private:
    symmetry_type m_sym; //!< Symmetry of target block tensor
    block_tensor_type &m_bt; //!< Target block tensor
    block_tensor_ctrl_type m_ctrl; //!< Control on target block tensor
    bool m_open; //!< Open state

public:
    /** \brief Constructs the operation
        \brief sym Symmetry of the target block tensor.
        \brief bt Target block tensor.
     **/
    bto_aux_copy(
        const symmetry_type &sym,
        block_tensor_type &bt);

    /** \brief Virtual destructor
     **/
    virtual ~bto_aux_copy();

    /** \brief Implements bto_stream_i::open(). Prepares the copy operation
     **/
    virtual void open();

    /** \brief Implements bto_stream_i::close()
     **/
    virtual void close();

    /** \brief Implements bto_stream_i::put(). Saves a block in the output
            block tensor
     **/
    virtual void put(
        const index<N> &idx,
        block_type &blk,
        const tensor_transf_type &tr);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_COPY_H
