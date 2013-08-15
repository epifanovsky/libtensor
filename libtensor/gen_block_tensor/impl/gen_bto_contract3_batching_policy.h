#ifndef LIBTENSOR_GEN_BTO_CONTRACT3_BATCHING_POLICY_H
#define LIBTENSOR_GEN_BTO_CONTRACT3_BATCHING_POLICY_H

#include <algorithm>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/batching_policy_base.h>

namespace libtensor {


/** \brief Preliminary batching policy class for contraction of three tensors

    TODO: Improve class and algorithms as soon as it is clear what the optimal
    values for batching sizes are

    \ingroup libtensor_gen_bto
 **/
template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
class gen_bto_contract3_batching_policy {
private:
    enum {
        NA = N1 + K1, //!< Rank of tensor A
        NB = N2 + K1 + K2, //!< Rank of tensor B
        NAB = N1 + N2 + K2, //!< Rank of intermediate tensor (A*B)
        NC = N3 + K2, //!< Rank of tensor C
        ND = N1 + N2 + N3 //!< Rank of result tensor (D)
    };

private:
    sequence<5, size_t> m_bsz; //!< Batch sizes


public:
    /** \brief Constructs the batching data
        \param contr1 First contraction
        \param contr2 Second contraction
        \param nblka Number of blocks in A
        \param nblkb Number of blocks in B
        \param nblkc Number of blocks in C
        \param nblkab Number of blocks in intermediate A * B
        \param nblkd Number of blocks in result
     **/
    gen_bto_contract3_batching_policy(
            const contraction2<N1, N2 + K2, K1> &contr1,
            const contraction2<N1 + N2, N3, K2> &contr2,
            size_t nblka, size_t nblkb, size_t nblkc,
            size_t nblkab, size_t nblkd);

    size_t get_bsz_a() { return m_bsz[0]; }
    size_t get_bsz_b() { return m_bsz[1]; }
    size_t get_bsz_c() { return m_bsz[2]; }
    size_t get_bsz_ab() { return m_bsz[3]; }
    size_t get_bsz_d() { return m_bsz[4]; }
};


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
gen_bto_contract3_batching_policy<N1, N2, N3, K1, K2>::
gen_bto_contract3_batching_policy(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    size_t nblka, size_t nblkb, size_t nblkc,
    size_t nblkab, size_t nblkd) {

    size_t batch_size = batching_policy_base::get_batch_size();
    //size_t nblktot = nblka + nblkb + nblkc + nblkab + nblkd;
    size_t bsza, bszb, bszc, bszab, bszd;
    size_t nbata, nbatb, nbatc, nbatab, nbatd;

    //bsza = std::max(batch_size * nblka / nblktot, size_t(1));
    //bszb = std::max(batch_size * nblkb / nblktot, size_t(1));
    //bszc = std::max(batch_size * nblkc / nblktot, size_t(1));
    //bszab = std::max(batch_size * nblkab / nblktot, size_t(1));
    //bszd = std::max(batch_size * nblkd / nblktot, size_t(1));
    bsza = std::max(std::min(batch_size / 3, nblka), size_t(1));
    bszb = std::max(std::min(batch_size / 3, nblkb), size_t(1));
    bszc = std::max(std::min(batch_size / 3, nblkc), size_t(1));
    bszab = std::max(std::min(batch_size / 3, nblkab), size_t(1));
    bszd = std::max(std::min(batch_size / 3, nblkd), size_t(1));

    nbata = (nblka + bsza - 1) / bsza;
    m_bsz[0] = (nbata > 0 ? (nblka + nbata - 1) / nbata : 1);
    nbatb = (nblkb + bszb - 1) / bszb;
    m_bsz[1] = (nbatb > 0 ? (nblkb + nbatb - 1) / nbatb : 1);
    nbatc = (nblkc + bszc - 1) / bszc;
    m_bsz[2] = (nbatc > 0 ? (nblkc + nbatc - 1) / nbatc : 1);
    nbatab = (nblkab + bszab - 1) / bszab;
    m_bsz[3] = (nbatab > 0 ? (nblkab + nbatab - 1) / nbatab : 1);
    nbatd = (nblkd + bszd - 1) / bszd;
    m_bsz[4] = (nbatd > 0 ? (nblkd + nbatd - 1) / nbatd : 1);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BATCHING_POLICY_H
