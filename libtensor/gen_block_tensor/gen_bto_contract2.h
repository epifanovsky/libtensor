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


/** \brief Computes one batch of the contraction of two block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    \sa gen_bto_contract2_basic

    \ingroup libtensor_block_tensor_btod
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
    typedef typename bti_traits::template wr_block_type<NC>::type wr_block_type;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First argument (A)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second argument (B)
    gen_bto_contract2_sym<N, M, K, Traits> m_symc; //!< Symmetry of the result

    dimensions<NA> m_bidimsa; //!< Block %index dims of A
    dimensions<NB> m_bidimsb; //!< Block %index dims of B
    dimensions<NC> m_bidimsc; //!< Block %index dims of the result
    assignment_schedule<NC, element_type> m_sch; //!< Assignment schedule

    size_t m_batch_size; //!< Batch size to use

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta Block %tensor A (first argument).
        \param btb Block %tensor B (second argument).
        \param batch_size Batch size.
    **/
    gen_bto_contract2(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        size_t batch_size = 4096);

    /** \brief Returns the block index space of the result
     **/
    const block_index_space<NC> &get_bis() const {

        return m_symc.get_bisc();
    }

    /** \brief Returns the symmetry of the result
     **/
    const symmetry<N + M, element_type> &get_symmetry() const {

        return m_symc.get_symc();
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
        const tensor_transf<NC, double> &trc,
        wr_block_type &blk);

private:
    void make_schedule();

    void align(const sequence<2 * (N + M + K), size_t> &conn,
        permutation<NA> &perma, permutation<NB> &permb,
        permutation<NC> &permc);
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_H
