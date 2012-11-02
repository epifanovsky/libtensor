#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H

#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


/** \brief Computes single blocks of the contraction of two block tensors
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Traits class.
    \tparam Timed Class for timings.

    This algorithm determines the list of required block contractions
    (\sa gen_bto_contract2_clst_builder) and uses it to compute the
    requested block.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class
    - \c template to_set_type<NX>::type -- Type of tensor operation to_set
    - \c template to_contract2_type<N, M, K>::type -- Type of tensor
            operation to_contract2
    - \c template to_contract2_type<N, M, K>::clst_optimize_type -- Type of
            contraction pair list optimizer (\sa gen_bto_contract2_clst_builder)

    \sa gen_bto_contract2

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_block : public timings<Timed>, public noncopyable {
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

    //! Type of read-only block (A)
    typedef typename bti_traits::template rd_block_type<NA>::type
        rd_block_a_type;

    //! Type of read-only block (B)
    typedef typename bti_traits::template rd_block_type<NB>::type
        rd_block_b_type;

    //! Type of write-only block (C)
    typedef typename bti_traits::template wr_block_type<NC>::type
        wr_block_c_type;

private:
    contraction2<N, M, K> m_contr; //!< Contraction
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta; //!< First block tensor (A)
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta2; //!< A with broken symmetry
    dimensions<NA> m_bidimsa; //!< Block index dims in A
    orbit_list<NA, element_type> m_ola; //!< List of orbits in A
    block_list<NA> m_bla; //!< List of non-zero blocks in A
    scalar_transf<element_type> m_ka; //!< Scalar transformation of A
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb; //!< Second block tensor (B)
    gen_block_tensor_rd_i<NB, bti_traits> &m_btb2; //!< B with broken symmetry
    dimensions<NB> m_bidimsb; //!< Block index dims in B
    orbit_list<NB, element_type> m_olb; //!< List of orbits in B
    block_list<NB> m_blb; //!< List of non-zero blocks in B
    scalar_transf<element_type> m_kb; //!< Scalar transformation of B
    dimensions<NC> m_bidimsc; //!< Block index dims in C
    scalar_transf<element_type> m_kc; //!< Scalar transformation of C
    bool m_use_broken_sym; //!< Whether to use broken symmetry

public:
    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta First block tensor (A).
        \param ka Scalar transform of A.
        \param btb Second block tensor (B).
        \param kb Scalar transform of B.
        \param bisc Block index space of result (C).
        \param kc Scalar transform of C.
     **/
    gen_bto_contract2_block(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const symmetry<NA, element_type> &syma,
        const block_list<NA> &bla,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const symmetry<NB, element_type> &symb,
        const block_list<NB> &blb,
        const scalar_transf<element_type> &kb,
        const block_index_space<NC> &bisc,
        const scalar_transf<element_type> &kc);

    /** \brief Initializes the contraction operation
        \param contr Contraction.
        \param bta First block tensor (A).
        \param bta2 A with broken symmetry.
        \param ka Scalar transform of A.
        \param btb Second block tensor (B).
        \param btb2 B with broken symmetry.
        \param kb Scalar transform of B.
        \param bisc Block index space of result (C).
        \param kc Scalar transform of C.
     **/
    gen_bto_contract2_block(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NA, bti_traits> &bta2,
        const symmetry<NA, element_type> &syma,
        const block_list<NA> &bla,
        const scalar_transf<element_type> &ka,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        gen_block_tensor_rd_i<NB, bti_traits> &btb2,
        const symmetry<NB, element_type> &symb,
        const block_list<NB> &blb,
        const scalar_transf<element_type> &kb,
        const block_index_space<NC> &bisc,
        const scalar_transf<element_type> &kc);

    void compute_block(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf<NC, element_type> &trc,
        wr_block_c_type &blkc);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_H
