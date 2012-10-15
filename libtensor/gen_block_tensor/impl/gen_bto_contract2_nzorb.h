#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_i.h>
#include "block_list.h"

namespace libtensor {


/** \brief Produces the list of non-zero canonical blocks that result from
        a contraction of two block tensors

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_contract2_nzorb : public timings<Timed>, public noncopyable {
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

private:
    contraction2<N, M, K> m_contr; //!< Contraction descriptor
    symmetry<NA, element_type> m_syma; //!< Symmetry of A
    symmetry<NB, element_type> m_symb; //!< Symmetry of B
    symmetry<NC, element_type> m_symc; //!< Symmetry of result (C)
    block_list<NA> m_blsta; //!< List of non-zero canonical blocks in A
    block_list<NB> m_blstb; //!< List of non-zero canonical blocks in B
    block_list<NC> m_blstc; //!< List of non-zero canonical blocks in C

public:
    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param bta First block tensor (A).
        \param btb Second block tensor (B).
        \param symc Symmetry of the result of the contraction (C).
     **/
    gen_bto_contract2_nzorb(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const symmetry<NC, element_type> &symc);

    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param syma Symmetry of first block tensor (A).
        \param scha Assignment schedule for A
        \param btb Second block tensor (B).
        \param symc Symmetry of the result of the contraction (C).
     **/
    gen_bto_contract2_nzorb(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const assignment_schedule<NA, element_type> &scha,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const symmetry<NC, element_type> &symc);

    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param bta First block tensor (A).
        \param symb Symmetry of second block tensor (B).
        \param schb Assignment schedule for B
        \param symc Symmetry of the result of the contraction (C).
     **/
    gen_bto_contract2_nzorb(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const symmetry<NB, element_type> &symb,
        const assignment_schedule<NB, element_type> &schb,
        const symmetry<NC, element_type> &symc);

    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param syma Symmetry of first block tensor (A).
        \param scha Assignment schedule for A
        \param symb Symmetry of second block tensor (B).
        \param schb Assignment schedule for B
        \param symc Symmetry of the result of the contraction (C).
     **/
    gen_bto_contract2_nzorb(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const assignment_schedule<NA, element_type> &scha,
        const symmetry<NB, element_type> &symb,
        const assignment_schedule<NB, element_type> &schb,
        const symmetry<NC, element_type> &symc);

    /** \brief Returns the list of non-zero canonical blocks
     **/
    const block_list<NC> &get_blst() const {
        return m_blstc;
    }

    /** \brief Builds the list
     **/
    void build();
};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_H
