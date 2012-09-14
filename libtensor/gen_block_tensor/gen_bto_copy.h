#ifndef LIBTENSOR_GEN_BTO_COPY_H
#define LIBTENSOR_GEN_BTO_COPY_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Copies a block tensor with an optional transformation
    \tparam N Tensor order.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    This algorithm prepares the block index space, symmetry and list of non-zero
    blocks to produce an exact copy of a block tensor. Blocks are copied
    in parallel to an output stream.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, typename Traits, typename Timed>
class gen_bto_copy : public timings<Timed>, public noncopyable {
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
    gen_block_tensor_rd_i<N, bti_traits> &m_bta; //!< Source block tensor (A)
    tensor_transf<N, element_type> m_tra; //!< Tensor transformation (A to B)
    block_index_space<N> m_bisb; //!< Block index space of B
    symmetry<N, element_type> m_symb; //!< Symmetry of B
    assignment_schedule<N, element_type> m_schb; //!< Non-zero list of B

public:
    /** \brief Initializes the copy operation
        \param bta Source block tensor.
        \param tra Transformation of the source tensor.
     **/
    gen_bto_copy(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<N> &get_bis() const {

        return m_bisb;
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {

        return m_symb;
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N, element_type> &get_schedule() const {

        return m_schb;
    }

    /** \brief Turns on synchronization on all arguments
     **/
    void sync_on();

    /** \brief Turns off synchronization on all arguments
     **/
    void sync_off();

    /** \brief Writes the blocks of the result to an output stream
        \param out Output stream.
     **/
    void perform(
        gen_block_stream_i<N, bti_traits> &out);

    /** \brief Writes a subset of blocks of the result to an output stream
        \param blst List of absolute indexes of canonical blocks to be computed.
        \param out Output stream.
     **/
    void perform(
        const std::vector<size_t> &blst,
        gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
     **/
    void compute_block(
        bool zero,
        wr_block_type &blkb,
        const index<N> &ib,
        const tensor_transf<N, element_type> &trb,
        const element_type &c);

private:
    void make_schedule();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_COPY_H
