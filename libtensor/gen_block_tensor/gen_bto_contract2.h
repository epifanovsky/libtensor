#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include "impl/gen_bto_contract2_sym.h"
#include "assignment_schedule.h"
#include "gen_block_stream_i.h"
#include "gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the contraction of two general block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Traits class for this block tensor operation.
    \tparam Timed Class name to identify timer with.

    This algorithm computes the contraction of two general block tensors
    in batches. It prepares block index space and symmetry
    (\sa gen_bto_contract2_sym), determines the optimal way of contraction
    (\sa gen_bto_contract2_align), and collects the tensor blocks which are
    to be computed in one batch. The calculation of each batch is then
    handed to gen_bto_contract2_batch.

    The batch-wise formation of the result is done as follows
    \f[
      C = \sum_{ij} A_i B_j \qquad
      A=\sum_i A_i \mbox{ and } B=\sum_i B_i
    \f]

    TODO: Improve the way the batches are determined.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template temp_block_tensor_type<NX>::type -- Type of temporary
            block tensors
    - \c template to_set_type<NX>::type -- Type of tensor operation to_set
    - \c template to_contract2_type<N, M, K>::type -- Type of tensor
            operation to_contract2
    - \c template to_contract2_type<N, M, K>::clst_optimize_type -- Type of
            contraction pair list optimizer (\sa gen_bto_contract2_clst_builder)

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2 : public timings<Timed>, public noncopyable {
private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

    //! Type of read-only block of A
    typedef typename bti_traits::template rd_block_type<NA>::type
            rd_block_a_type;

    //! Type of read-only block of A
    typedef typename bti_traits::template rd_block_type<NB>::type
            rd_block_b_type;

    //! Type of write-only block
    typedef typename bti_traits::template wr_block_type<NC>::type
            wr_block_type;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First argument (A)
    scalar_transf<element_type> m_ka; //!< Scalar transform of A.
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second argument (B)
    scalar_transf<element_type> m_kb; //!< Scalar transform of B.
    scalar_transf<element_type> m_kc; //!< Scalar transform of the result.
    gen_bto_contract2_sym<N, M, K, Traits> m_symc; //!< Symmetry of the result
    assignment_schedule<NC, element_type> m_sch; //!< Assignment schedule

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param ka Scalar transform of A.
        \param btb Block %tensor B (second argument).
        \param kb Scalar transform of B.
        \param kc Scalar transform of the result (C).
    **/
    gen_bto_contract2(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const scalar_transf<element_type> &kb,
        const scalar_transf<element_type> &kc);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<NC> &get_bis() const {

        return m_symc.get_bis();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N + M, element_type> &get_symmetry() const {

        return m_symc.get_symmetry();
    }

    /** \brief Returns the list of canonical non-zero blocks of the result
     **/
    const assignment_schedule<N + M, element_type> &get_schedule() const {

        return m_sch;
    }

    /** \brief Computes the contraction into an output stream
     **/
    void perform(gen_block_stream_i<NC, bti_traits> &out);


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
        wr_block_type &blk);

private:
    void make_schedule();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_H
