#ifndef LIBTENSOR_BTO_CONTRACT2_BIS_H
#define LIBTENSOR_BTO_CONTRACT2_BIS_H

#include <libtensor/core/block_index_space.h>
#include <libtensor/dense_tensor/to_contract2_dims.h>

namespace libtensor {


/** \brief Computes the block index space of the result of a contraction

    Given the block index spaces of the arguments of a contraction and
    the contraction descriptor, this class builds the block index space of
    the result.

    \ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class bto_contract2_bis {
private:
    to_contract2_dims<N, M, K> m_dimsc; //!< Builder of dimensions of C
    block_index_space<N + M> m_bisc; //!< Block index space of result

public:
    /** \brief Computes the block index space of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param bisb Block index space of B.
     **/
    bto_contract2_bis(const contraction2<N, M, K> &contr,
        const block_index_space<N + K> &bisa,
        const block_index_space<M + K> &bisb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bisc() const {
        return m_bisc;
    }

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_bis(const bto_contract2_bis&);

};


/** \brief Computes the block index space of the result of a contraction
        (specialized for no contracted indexes, i.e. direct product)

    \ingroup libtensor
 **/
template<size_t N, size_t M>
class bto_contract2_bis<N, M, 0> {
private:
    to_contract2_dims<N, M, 0> m_dimsc; //!< Builder of dimensions of C
    block_index_space<N + M> m_bisc; //!< Block index space of result

public:
    /** \brief Computes the block index space of C
        \param contr Contraction.
        \param bisa Block index space of A.
        \param bisb Block index space of B.
     **/
    bto_contract2_bis(const contraction2<N, M, 0> &contr,
        const block_index_space<N> &bisa,
        const block_index_space<M> &bisb);

    /** \brief Returns the block index space of C
     **/
    const block_index_space<N + M> &get_bisc() const {
        return m_bisc;
    }

private:
    /** \brief Private copy constructor
     **/
    bto_contract2_bis(const bto_contract2_bis&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_BIS_H
