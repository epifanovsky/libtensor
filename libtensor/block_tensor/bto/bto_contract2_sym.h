#ifndef LIBTENSOR_BTO_CONTRACT2_SYM_H
#define LIBTENSOR_BTO_CONTRACT2_SYM_H

#include <libtensor/core/symmetry.h>
#include "bto_contract2_bis.h"

namespace libtensor {


/** \brief Computes the symmetry of the result of a contraction

    Given the spaces and symmetries of the arguments of a contraction and
    the contraction descriptor, this class builds the symmetry of the result.

    \ingroup libtensor
 **/
template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_sym {
private:
    bto_contract2_bis<N, M, K> m_bisc; //!< Builder of block index space of C
    symmetry<N + M, T> m_symc; //!< Symmetry of result C

public:
    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param syma Symmetry of A.
        \param bisb Block index space of B.
        \param symb Symmetry of B.
     **/
    bto_contract2_sym(const contraction2<N, M, K> &contr,
        const block_index_space<N + K> &bisa, const symmetry<N + K, T> &syma,
        const block_index_space<M + K> &bisb, const symmetry<M + K, T> &symb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bisc() const {
        return m_bisc.get_bisc();
    }

    /** \brief Returns the symmetry of C
     **/
    const symmetry<N + M, T> &get_symc() const {
        return m_symc;
    }

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_sym(const bto_contract2_sym&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_SYM_H
