#ifndef LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_H
#define LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_H

#include "gen_bto_contract2_sym.h"

namespace libtensor {


/** \brief Computes the symmetry of the result of a contraction
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.
    \tparam Traits Traits class.

    Given the spaces and symmetries of the arguments of a contraction and
    the contraction descriptor, this class builds the symmetry of the result.

    The traits class has to provide definitions for
    - \c element_type -- Type of data elements
    - \c bti_traits -- Type of block tensor interface traits class

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_symcontract2_sym : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

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
    gen_bto_contract2_sym<N, M, K, Traits> m_symbld; //!< Builder of contraction symmetry
    symmetry<NC, element_type> m_sym; //!< Symmetry of result C

public:
    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bta Block tensor A.
        \param btb Block tensor B.
        \param perm Symmetrization permutation.
        \param symm Symmetrization (true)/antisymmetrization (false).
     **/
    gen_bto_symcontract2_sym(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const permutation<NC> &perm,
        bool symm);

    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param syma Symmetry of A.
        \param symb Symmetry of B.
        \param perm Symmetrization permutation.
        \param symm Symmetrization (true)/antisymmetrization (false).
     **/
    gen_bto_symcontract2_sym(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const permutation<NC> &perm,
        bool symm);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bis() const {
        return m_symbld.get_bis();
    }

    /** \brief Returns the symmetry of C
     **/
    const symmetry<N + M, element_type> &get_symmetry() const {
        return m_sym;
    }

    /** \brief Returns the pre-symmetrized symmetry of C
     **/
    const symmetry<N + M, element_type> &get_symmetry0() const {
        return m_symbld.get_symmetry();
    }

private:
    void make_symmetry(
        const permutation<NC> &perm,
        bool symm);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_SYMCONTRACT2_SYM_H
