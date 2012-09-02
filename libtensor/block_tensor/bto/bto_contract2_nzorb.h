#ifndef LIBTENSOR_BTO_CONTRACT2_NZORB_H
#define LIBTENSOR_BTO_CONTRACT2_NZORB_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/tod/contraction2.h>

namespace libtensor {


/** \brief Produces the list of non-zero canonical blocks that result from
        a contraction of two block tensors

    \ingroup libtensor_block_tensor
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb : public timings< bto_contract2_nzorb<N, M, K, T> > {
public:
    static const char *k_clazz; //!< Class name

private:
    contraction2<N, M, K> m_contr; //!< Contraction descriptor
    block_tensor_i<N + K, T> &m_bta; //!< First block tensor (A)
    block_tensor_i<M + K, T> &m_btb; //!< Second block tensor (B)
    symmetry<N + M, T> m_symc; //!< Symmetry of result (C)
    std::vector<size_t> m_blst; //!< List of non-zero canonical blocks

public:
    /** \brief Initializes the operation
        \param contr Contraction descriptor.
        \param bta First block tensor (A).
        \param btb Second block tensor (B).
        \param symc Symmetry of the result of the contraction (C).
     **/
    bto_contract2_nzorb(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const symmetry<N + M, T> &symc);

    /** \brief Returns the list of non-zero canonical blocks
     **/
    const std::vector<size_t> &get_blst() const {
        return m_blst;
    }

    /** \brief Builds the list
     **/
    void build();

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_NZORB_H
