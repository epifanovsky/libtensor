#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_SYM_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_SYM_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/symmetry.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_i.h>
#include "gen_bto_contract2_bis.h"

namespace libtensor {


/** \brief Computes the symmetry of the result of a contraction

    Given the spaces and symmetries of the arguments of a contraction and
    the contraction descriptor, this class builds the symmetry of the result.

    \ingroup libtensor
 **/
template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_sym : public noncopyable {
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
    gen_bto_contract2_bis<N, M, K> m_bis; //!< Builder of block index space of C
    symmetry<NC, element_type> m_sym; //!< Symmetry of result C

public:
    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bta Block tensor A.
        \param btb Block tensor B.
     **/
    gen_bto_contract2_sym(
        const contraction2<N, M, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb);

    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param syma Symmetry of A.
        \param symb Symmetry of B.
     **/
    gen_bto_contract2_sym(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bis() const {
        return m_bis.get_bis();
    }

    /** \brief Returns the symmetry of C
     **/
    const symmetry<N + M, element_type> &get_symmetry() const {
        return m_sym;
    }

private:
    void make_symmetry(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb);

};


/** \brief Computes the symmetry of the result of a contraction (specialized
        for same-order tensors)

    Given the spaces and symmetries of the arguments of a contraction and
    the contraction descriptor, this class builds the symmetry of the result.

    \ingroup libtensor
 **/
template<size_t N, size_t K, typename Traits>
class gen_bto_contract2_sym<N, N, K, Traits> : public noncopyable {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = N + K, //!< Order of second argument (B)
        NC = N + N  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_bis<N, N, K> m_bis; //!< Bldr of block index space of C
    symmetry<NC, element_type> m_sym; //!< Symmetry of result C

public:
    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bta Block tensor A.
        \param btb Block tensor B.
     **/
    gen_bto_contract2_sym(
        const contraction2<N, N, K> &contr,
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        gen_block_tensor_rd_i<NB, bti_traits> &btb);

    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param syma Symmetry of A.
        \param bisb Block index space of B.
        \param symb Symmetry of B.
        \param self Contraction of tensor with itself (if true).
     **/
    gen_bto_contract2_sym(
        const contraction2<N, N, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        bool self = false);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + N> &get_bis() const {
        return m_bis.get_bis();
    }

    /** \brief Returns the symmetry of C
     **/
    const symmetry<N + N, element_type> &get_symmetry() const {
        return m_sym;
    }

private:
    void make_symmetry(
        const contraction2<N, N, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        bool self);

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_SYM_H
