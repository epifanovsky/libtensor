#ifndef LIBTENSOR_BTOD_CONTRACT3_H
#define LIBTENSOR_BTOD_CONTRACT3_H

#include <vector>
#include <libtensor/tod/contraction2.h>
#include <libtensor/core/block_tensor_i.h>
#include <libtensor/btod/btod_contract2.h>

namespace libtensor {


/** \brief Contracts a train of three tensors
    \tparam N1 Order of first tensor less first contraction degree.
    \tparam N2 Order of second tensor less total contraction degree.
    \tparam N3 Order of third tensor less second contraction degree.
    \tparam K1 First contraction degree.
    \tparam K2 Second contraction degree.

    This algorithm computes the contraction of three linearly connected tensors.
    In this train of tensors, the first tensor may only be connected (have
    shared inner indexes) with the second, and the second tensor may only be
    connected with the third. No connections between the first and the third
    tensors are allowed.

    The contraction is performed as follows. The first tensor is contracted
    with the second tensor to form an intermediate, which is then contracted
    with the third tensor to yield the final result.

    The formation of the intermediate is done in batches:
    \f[
        ABC = A(B_1 + B_2 + \dots + B_n)C = \sum_{i=1}^n (AB_i)C \qquad
        B = \sum_{i=1}^n B_i
    \f]

    \ingroup libtensor_block_tensor
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
class btod_contract3 {
public:
    static const char *k_clazz; //!< Class name

private:
    contraction2<N1, N2 + K2, K1> m_contr1; //!< First contraction
    contraction2<N1 + N2, N3, K2> m_contr2; //!< Second contraction
    block_tensor_i<N1 + K1, double> &m_bta; //!< First argument (A)
    block_tensor_i<N2 + K1 + K2, double> &m_btb; //!< Second argument (B)
    block_tensor_i<N3 + K2, double> &m_btc; //!< Third argument (C)

public:
    /** \brief Initializes the contraction
        \param contr1 First contraction (A with B).
        \param contr2 Second contraction (AB with C).
        \param bta First tensor argument (A).
        \param btb Second tensor argument (B).
        \param btc Third tensor argument (C).
     **/
    btod_contract3(
        const contraction2<N1, N2 + K2, K1> &contr1,
        const contraction2<N1 + N2, N3, K2> &contr2,
        block_tensor_i<N1 + K1, double> &bta,
        block_tensor_i<N2 + K1 + K2, double> &btb,
        block_tensor_i<N3 + K2, double> &btc);

    /** \brief Computes the contraction
     **/
    void perform(block_tensor_i<N1 + N2 + N3, double> &btd);

private:
    void compute_batch_ab(
        btod_contract2<N1, N2 + K2, K1> &contr,
        const std::vector<size_t> &blst,
        block_tensor_i<N1 + N2 + K2, double> &btab);

private:
    btod_contract3(const btod_contract3&);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT3_H

