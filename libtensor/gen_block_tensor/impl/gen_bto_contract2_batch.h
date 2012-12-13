#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BATCH_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BATCH_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include "../gen_block_stream_i.h"
#include "../gen_block_tensor_i.h"

namespace libtensor {


/** \brief Computes the requested batches of the contraction of two tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Block tensor operation traits.
    \tparam Timed Timed implementation.

    Computes the requested batches of the contraction of two block tensors
    in parallel (if applicable).

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_batch : public timings<Timed>, public noncopyable {
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
    permutation<NA> m_perma; //!< Permutation of A
    scalar_transf<element_type> m_ka; //!< Scalar transformation of A
    const std::vector<size_t> &m_batcha; //!< List of blocks in A
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    permutation<NB> m_permb; //!< Permutation of B
    scalar_transf<element_type> m_kb; //!< Scalar transformation of B
    const std::vector<size_t> &m_batchb; //!< List of blocks in B
    block_index_space<NC> m_bisc; //!< Block index space of result (C)
    scalar_transf<element_type> m_kc; //!< Scalar transformation of C

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta First block tensor (A).
        \param perma Permutation of A.
        \param ka Scalar transform of A.
        \param batcha List of blocks in A.
        \param btb Second block tensor (B).
        \param permb Permutation of B.
        \param kb Scalar transform of B.
        \param batchb List of blocks in B.
        \param bisc Block index space of result (C).
        \param kc Scalar transform of C.
     **/
    gen_bto_contract2_batch(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const permutation<NA> &perma,
        const scalar_transf<element_type> &ka,
        const std::vector<size_t> &batcha,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const permutation<NB> &permb,
        const scalar_transf<element_type> &kb,
        const std::vector<size_t> &batchb,
        const block_index_space<NC> &bisc,
        const scalar_transf<element_type> &kc);

    /** \brief Computes and writes the blocks of the result to an output stream
        \param blst List of absolute indexes of canonical blocks to be computed.
        \param out Output stream.
     **/
    void perform(
        const std::vector<size_t> &blst,
        gen_block_stream_i<NC, bti_traits> &out);
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BASIC_H
