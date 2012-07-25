#ifndef LIBTENSOR_BTO_CONTRACT2_SYM_H
#define LIBTENSOR_BTO_CONTRACT2_SYM_H

#include <libtensor/core/block_tensor_i.h>
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
        \param bta Block tensor A.
        \param btb Block tensor B.
     **/
    bto_contract2_sym(const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta, block_tensor_i<M + K, T> &btb);

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
    void make_symmetry(const contraction2<N, M, K> &contr,
        const block_index_space<N + K> &bisa, const symmetry<N + K, T> &syma,
        const block_index_space<M + K> &bisb, const symmetry<M + K, T> &symb);

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_sym(const bto_contract2_sym&);

};


/** \brief Computes the symmetry of the result of a contraction (specialized
        for same-order tensors)

    Given the spaces and symmetries of the arguments of a contraction and
    the contraction descriptor, this class builds the symmetry of the result.

    \ingroup libtensor
 **/
template<size_t N, size_t K, typename T>
class bto_contract2_sym<N, N, K, T> {
private:
    bto_contract2_bis<N, N, K> m_bisc; //!< Builder of block index space of C
    symmetry<N + N, T> m_symc; //!< Symmetry of result C

public:
    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bta Block tensor A.
        \param btb Block tensor B.
     **/
    bto_contract2_sym(const contraction2<N, N, K> &contr,
        block_tensor_i<N + K, T> &bta, block_tensor_i<N + K, T> &btb);

    /** \brief Computes the symmetry of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param syma Symmetry of A.
        \param bisb Block index space of B.
        \param symb Symmetry of B.
        \param self Contraction of tensor with itself (if true).
     **/
    bto_contract2_sym(const contraction2<N, N, K> &contr,
        const block_index_space<N + K> &bisa, const symmetry<N + K, T> &syma,
        const block_index_space<N + K> &bisb, const symmetry<N + K, T> &symb,
        bool self = false);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + N> &get_bisc() const {
        return m_bisc.get_bisc();
    }

    /** \brief Returns the symmetry of C
     **/
    const symmetry<N + N, T> &get_symc() const {
        return m_symc;
    }

private:
    void make_symmetry(const contraction2<N, N, K> &contr,
        const block_index_space<N + K> &bisa, const symmetry<N + K, T> &syma,
        const block_index_space<N + K> &bisb, const symmetry<N + K, T> &symb,
        bool self);

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_sym(const bto_contract2_sym&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_SYM_H
