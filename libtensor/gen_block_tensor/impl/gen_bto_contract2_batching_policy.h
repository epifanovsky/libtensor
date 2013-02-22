#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BATCHING_POLICY_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BATCHING_POLICY_H

#include <libtensor/core/contraction2.h>

namespace libtensor {


/** \brief Preliminary batching policy class for contraction of two tensors

    TODO: Improve class and algorithms as soon as it is clear what the optimal
    values for batching sizes are

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K>
class gen_bto_contract2_batching_policy {
private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M //!< Order of result (C)
    };

private:
    sequence<3, size_t> m_bsz; //!< Batch sizes


public:
    /** \brief Constructs the batching data
        \param contr Contraction
        \param nblka Number of blocks in A
        \param nblkb Number of blocks in B
        \param nblkc Number of blocks in result
     **/
    gen_bto_contract2_batching_policy(const contraction2<N, M, K> &contr,
            size_t nblka, size_t nblkb, size_t nblkc);

    size_t get_bsz_a() { return m_bsz[0]; }
    size_t get_bsz_b() { return m_bsz[1]; }
    size_t get_bsz_c() { return m_bsz[2]; }
};


template<size_t N, size_t M, size_t K>
gen_bto_contract2_batching_policy<N, M, K>::
gen_bto_contract2_batching_policy(const contraction2<N, M, K> &contr,
        size_t nblka, size_t nblkb, size_t nblkc) {

    size_t batch_size = 1024;
    size_t nbata, nbatb, nbatc;

    nbata = (nblka + batch_size - 1) / batch_size;
    m_bsz[0] = (nbata > 0 ? (nblka + nbata - 1) / nbata : 1);
    nbatb = (nblkb + batch_size - 1) / batch_size;
    m_bsz[1] = (nbatb > 0 ? (nblkb + nbatb - 1) / nbatb : 1);
    nbatc = (nblkc + batch_size - 1) / batch_size;
    m_bsz[2] = (nbatc > 0 ? (nblkc + nbatc - 1) / nbatc : 1);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BATCHING_POLICY_H
