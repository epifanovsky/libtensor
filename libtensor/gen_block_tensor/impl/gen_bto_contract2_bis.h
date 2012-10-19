#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BIS_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BIS_H

#include <libtensor/core/block_index_space.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>

namespace libtensor {


/** \brief Computes the block index space of the result of a contraction
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    Given the block index spaces of the arguments of a contraction and
    the contraction descriptor, this class builds the block index space of
    the result.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K>
class gen_bto_contract2_bis : public noncopyable {
private:
    to_contract2_dims<N, M, K> m_dims; //!< Builder of dimensions of C
    block_index_space<N + M> m_bis; //!< Block index space of result

public:
    /** \brief Computes the block index space of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param bisb Block index space of B.
     **/
    gen_bto_contract2_bis(
        const contraction2<N, M, K> &contr,
        const block_index_space<N + K> &bisa,
        const block_index_space<M + K> &bisb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bis() const {
        return m_bis;
    }

};


/** \brief Computes the block index space of the result of a contraction
        (specialized for no contracted indexes, i.e. direct product)
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M>
class gen_bto_contract2_bis<N, M, 0> : public noncopyable {
private:
    to_contract2_dims<N, M, 0> m_dims; //!< Builder of dimensions of C
    block_index_space<N + M> m_bis; //!< Block index space of result

public:
    /** \brief Computes the block index space of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param bisb Block index space of B.
     **/
    gen_bto_contract2_bis(
        const contraction2<N, M, 0> &contr,
        const block_index_space<N> &bisa,
        const block_index_space<M> &bisb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bis() const {
        return m_bis;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BIS_H
