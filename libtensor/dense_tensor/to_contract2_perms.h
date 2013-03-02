#ifndef LIBTENSOR_TO_CONTRACT2_PERMS_H
#define LIBTENSOR_TO_CONTRACT2_PERMS_H

#include <libtensor/core/contraction2.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/permutation.h>

namespace libtensor {


/** \brief Computes the required permutations for given contraction operations.
 * Takes contraction operation and input ant output tensors dimensions as input
 * and gives permutations for each of the three tensors as output.
 * Contractions are done through the GEMM matrix multiplication. In order to use
 * GEMM summation and non-summation indices have to be separated in two groups.
 *
 * This problem exists only for tensors with more then 3 dimensions and
 * for contractions with contraction degree more then 1.
 *
 * \tparam N Order of the first %tensor (a) less the contraction degree
 * \tparam M Order of the second %tensor (b) less the contraction degree
 * \tparam K Contraction degree (the number of indexes over which the
 *         tensors are contracted)
 *
 *

    \ingroup libtensor
 **/
template<size_t N, size_t M, size_t K>
class to_contract2_perms {
private:
    enum {
        k_ordera = N + K, //!< Order of %tensor a
        k_orderb = M + K, //!< Order of %tensor b
        k_orderc = N + M, //!< Order of %tensor c
        k_totidx = N + M + K, //!< Total number of indexes
        k_maxconn = 2 * k_totidx, //!< Total number of indexes
    };

    permutation<k_ordera> m_perma; //!< Permutation of the first input %tensor (a)
    permutation<k_orderb> m_permb; //!< Permutation of the second input %tensor (b)
    permutation<k_orderc> m_permc; //!< Permutation of the output %tensor (c)


public:
    /** \brief Computes the permutations of tensors
        \param contr Contraction.
        \param dimsa Dimensions of A.
        \param dimsb Dimensions of B.
     **/
    to_contract2_perms(const contraction2<N, M, K> &contr,
        const dimensions<N + K> &dimsa, const dimensions<M + K> &dimsb, const dimensions<N + M> &dimsc)
        //m_perma(make_perma(contr, dimsa, dimsb)) //????
    {
        make_perms(contr, dimsa, dimsb, dimsc);
    }

    /** \brief Returns the permutation of A
     **/
    const permutation<k_ordera> &get_perma() const {
        return m_perma;
    }

    /** \brief Returns the permutation of B
     **/
    const permutation<k_orderb> &get_permb() const {
        return m_permb;
    }

    /** \brief Returns the permutation of C
     **/
    const permutation<k_orderc> &get_permc() const {
        return m_permc;
    }

private:
    void make_perms(const contraction2<N, M, K> &contr,
        const dimensions<N + K> &dimsa, const dimensions<M + K> &dimsb, const dimensions<N + M> &dimsc);

private:
    /** \brief Private copy constructor
     **/
    to_contract2_perms(const to_contract2_perms&);

template<size_t A, size_t B>
    bool does_permute_first(const dimensions<A> &dimsa,
            const dimensions<B> &dimsb, size_t perm_indexa, size_t perm_indexb, size_t permute_zone_a, size_t permute_zone_b);
    template<size_t A>
        size_t get_permute_cost(const dimensions<A> &dimsa,    size_t perm_indexa);

    void permute_conn(sequence<k_maxconn, size_t> &conn, size_t i, size_t j);

};


} // namespace libtensor

#endif // LIBTENSOR_TO_CONTRACT2_PERMS_H
