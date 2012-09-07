#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/block_tensor/bto/bto_contract2_sym.h>
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the contraction of two tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    This algorithm prepares the block index space, symmetry and list of non-zero
    blocks, as well as the result of the contraction of two block tensors.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2 : public timings<Timed>, public noncopyable {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

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
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First block tensor (A)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    bto_contract2_sym<N, M, K, element_type> m_symc; //!< Symmetry of result (C)

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta First block tensor (A).
        \param btb Second block tensor (B).
     **/
    gen_bto_contract2(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<NC> &get_bis() const {

        return m_symc.get_bisc();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N, element_type> &get_symmetry() const {

        return m_symc.get_symc();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
//    const assignment_schedule<N, element_type> &get_schedule() const {
//
//        return m_schb;
//    }

    /** \brief Turns on synchronization on all arguments
     **/
    void sync_on();

    /** \brief Turns off synchronization on all arguments
     **/
    void sync_off();

    /** \brief Writes the blocks of the result to an output stream
        \param out Output stream.
     **/
//    void perform(gen_block_stream_i<N, bti_traits> &out);

    /** \brief Computes one block of the result
     **/
//    void compute_block(
//        bool zero,
//        wr_block_type &blkb,
//        const index<N> &ib,
//        const tensor_transf<N, element_type> &trb,
//        const element_type &c);

    /** \brief Computes and writes the blocks of the result to an output stream
        \param blst List of absolute indexes of canonical blocks to be computed.
        \param out Output stream.
     **/
    void perform(
        const std::vector<size_t> &blst,
        gen_block_stream_i<NC, bti_traits> &out);

    /** \brief Computes one block of the result and writes it to a tensor
        \param zero Whether to zero out the contents of output before adding
            the contraction
        \param idxc Index of the computed block, must be a canonical block in
            the output tensor's symmetry
        \param trc Transformation to be applied to the computed block.
        \param[out] blkc Output tensor.
     */
    void compute_block(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf<NC, element_type> &trc,
        wr_block_type &blkc);

private:
    void make_schedule();

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_H
