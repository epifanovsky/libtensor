#ifndef LIBTENSOR_TO_EWMULT2_H
#define LIBTENSOR_TO_EWMULT2_H

#include <libtensor/timings.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief General element-wise multiplication of two tensors
    \tparam N First argument's %tensor order less the number of shared
        indexes.
    \tparam M Second argument's %tensor order less the number of shared
        indexes.
    \tparam K Number of shared indexes.

    This operation computes the generalized element-wise (Hadamard) product
    of two tensors. It takes two arguments and performs the following
    $$ c_{ij\cdots mn\cdots pq\cdots} =
        a_{ij\cdots pq\cdots} b_{mn\cdots pq\cdots} $$
    Both arguments have $K > 0$ shared indexes $pq\cdots$. In addition,
    the first argument has $N \ge 0$ and the second argument has $M \ge 0$
    extra indexes that are not subject to the operation. The result
    has $N+M+K$ indexes. $N$, $M$ and $K$ are the template parameters of
    the operation.

    The arguments A and B may be given in a permuted form. In this case
    permutations should be specified upon construction. The permutations for
    A and B rearrange the indexes to the standard form: $ij\cdots pq\cdots$
    and $mn\cdots pq\cdots$, respectively. The permutation for C transforms
    the standard index ordering $ij\cdots mn\cdots pq\cdots$ to the desired
    form.

    The output tensor C specified upon calling perform() must agree in
    dimensions with the input tensors A and B taking all permutations into
    account. A and B themselves must agree in the dimensions of the shared
    indexes. Any disagreement will raise bad_dimensions.

    \sa to_contract2

    \ingroup libtensor_dense_tensor_to
 **/
template<size_t N, size_t M, size_t K, typename T>
class to_ewmult2 :
    public timings< to_ewmult2<N, M, K, T> >,
    public noncopyable {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        k_ordera = N + K, //!< Order of first argument (A)
        k_orderb = M + K, //!< Order of second argument (B)
        k_orderc = N + M + K //!< Order of result (C)
    };

public:
    typedef tensor_transf<k_orderc, T> tensor_transf_type;

private:
    dense_tensor_rd_i<k_ordera, T> &m_ta; //!< First argument (A)
    permutation<k_ordera> m_perma; //!< Permutation of first argument (A)
    dense_tensor_rd_i<k_orderb, T> &m_tb; //!< Second argument (B)
    permutation<k_orderb> m_permb; //!< Permutation of second argument (B)
    permutation<k_orderc> m_permc; //!< Permutation of result (C)
    T m_d; //!< Scaling coefficient
    dimensions<k_orderc> m_dimsc; //!< Result dimensions

public:
    /** \brief Initializes the operation
        \param ta First argument (A).
        \param tra Tensor transformation of A.
        \param tb Second argument (B).
        \param trb Tensor transformation of B.
        \param trc Tensor transformation of result (C).
     **/
    to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
        const tensor_transf<k_ordera, T> &tra,
        dense_tensor_rd_i<k_orderb, T> &tb,
        const tensor_transf<k_orderb, T> &trb,
        const tensor_transf_type &trc = tensor_transf_type());

    /** \brief Initializes the operation
        \param ta First argument (A).
        \param tb Second argument (B).
        \param d Scaling coefficient.
     **/
    to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
        dense_tensor_rd_i<k_orderb, T> &tb, T d = 1.0);

    /** \brief Initializes the operation
        \param ta First argument (A).
        \param perma Permutation of A.
        \param tb Second argument (B).
        \param permb Permutation of B.
        \param permc Permutation of result (C).
        \param d Scaling coefficient.
     **/
    to_ewmult2(dense_tensor_rd_i<k_ordera, T> &ta,
        const permutation<k_ordera> &perma,
        dense_tensor_rd_i<k_orderb, T> &tb,
        const permutation<k_orderb> &permb,
        const permutation<k_orderc> &permc, T d = 1.0);

    /** \brief Destructor
     **/
    ~to_ewmult2();

    /** \brief Prefetches the arguments
     **/
    void prefetch();

    /** \brief Performs the operation
        \param zero Zero output before computing.
        \param tc Output tensor C.
     **/
    void perform(bool zero, dense_tensor_wr_i<k_orderc, T> &tc);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_EWMULT2_H
