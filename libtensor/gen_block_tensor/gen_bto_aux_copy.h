#ifndef LIBTENSOR_GEN_BTO_AUX_COPY_H
#define LIBTENSOR_GEN_BTO_AUX_COPY_H

#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"
#include "gen_block_tensor_ctrl.h"

namespace libtensor {


/** \brief Saves blocks into a block tensor (auxiliary operation)
    \tparam N Tensor order.
    \tparam BtiTraits Block tensor operation traits structure.

    This auxiliary block tensor operation accepts blocks and saves them into
    a given block tensor object. Upon initialization via open(), the operation
    zeroes out the target block tensor and installs the specified symmetry.
    Each put() will copy the block with the given index. If put() is called
    multiple times with the same block index, only the last one will be saved.
    The blocks must be canonical.

    \sa gen_block_stream_i

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits>
class gen_bto_aux_copy :
    public gen_block_stream_i<N, typename Traits::bti_traits> {

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;

private:
    symmetry<N, element_type> m_sym; //!< Symmetry of target block tensor
    gen_block_tensor_wr_i<N, bti_traits> &m_bt; //!< Target block tensor
    gen_block_tensor_wr_ctrl<N, bti_traits> m_ctrl; //!< Block tensor control
    bool m_open; //!< Open state

public:
    /** \brief Constructs the operation
        \brief sym Symmetry of the target block tensor.
        \brief bt Target block tensor.
     **/
    gen_bto_aux_copy(
        const symmetry<N, element_type> &sym,
        gen_block_tensor_wr_i<N, bti_traits> &bt);

    /** \brief Virtual destructor
     **/
    virtual ~gen_bto_aux_copy();

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
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_AUX_COPY_H
