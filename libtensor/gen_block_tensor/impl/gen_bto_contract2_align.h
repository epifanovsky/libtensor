#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_ALIGN_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_ALIGN_H

#include <libtensor/core/contraction2.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/permutation_builder.h>

namespace libtensor {


/** \brief Computes optimal contraction scheme
    \tparam N Order of first tensor less degree of contraction.
    \tparam M Order of second tensor less degree of contraction.
    \tparam K Order of contraction.

    Constructs the permutations of the tensors in a contraction such that
    the latter can be performed as matrix-matrix multiplication.

    \ingroup libtensor_gen_bto
 **/
template<size_t N, size_t M, size_t K>
class gen_bto_contract2_align : public noncopyable {
private:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of the result (C)
    };
private:
    contraction2<N, M, K> m_contr; //!< Contraction descriptor
    permutation<NA> m_perma; //!< Permutation of first argument (A)
    permutation<NB> m_permb; //!< Permutation of second argument (B)
    permutation<NC> m_permc; //!< Permutation of result (C)

public:
    /** \brief Initialize the permutations
        \param contr Contraction descriptor
     **/
    gen_bto_contract2_align(const contraction2<N, M, K> &contr);

    /** \brief Return permutation of first argument (A)
     **/
    const permutation<N + K> &get_perma() { return m_perma; }

    /** \brief Return permutation of second argument (B)
     **/
    const permutation<M + K> &get_permb() { return m_permb; }

    /** \brief Return permutation of the result (C)
     **/
    const permutation<N + M> &get_permc() { return m_permc; }

private:
    void build();
};


template<size_t N, size_t M, size_t K>
gen_bto_contract2_align<N, M, K>::gen_bto_contract2_align(
        const contraction2<N, M, K> &contr) : m_contr(contr) {

    build();
}


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_align<N, M, K>::build() {

    const sequence<2 * (N + M + K), size_t> &conn = m_contr.get_conn();

    //  This algorithm reorders indexes in A, B, C so that the whole contraction
    //  can be done in a single matrix multiplication.
    //  Returned permutations perma, permb, permc need to be applied to
    //  the indexes of A, B, and C to get the matricized form.

    //  Numbering scheme:
    //  0     .. N - 1         -- outer indexes from A
    //  N     .. N + M - 1     -- outer indexes from B
    //  N + M .. N + M + K - 1 -- inner indexes

    size_t ioa = 0, iob = N, ii = NC;

    sequence<NA, size_t> idxa1(0), idxa2(0);
    sequence<NB, size_t> idxb1(0), idxb2(0);
    sequence<NC, size_t> idxc1(0), idxc2(0);

    //  Build initial index ordering

    for(size_t i = 0; i < NC; i++) {
        size_t j = conn[i] - NC;
        if(j < NA) {
            idxc1[i] = ioa;
            idxa1[j] = ioa;
            ioa++;
        } else {
            j -= NA;
            idxc1[i] = iob;
            idxb1[j] = iob;
            iob++;
        }
    }
    for(size_t i = 0; i < NA; i++) {
        if(conn[NC + i] < NC) continue;
        size_t j = conn[NC + i] - NC - NA;
        idxa1[i] = ii;
        idxb1[j] = ii;
        ii++;
    }

    //  Build matricized index ordering

    size_t iai, iao, ibi, ibo, ica, icb;
    if(idxa1[NA - 1] >= NC) {
        //  Last index in A is an inner index
        iai = NA; iao = N;
    } else {
        //  Last index in A is an outer index
        iai = K; iao = NA;
    }
    if(idxb1[NB - 1] >= NC) {
        //  Last index in B is an inner index
        ibi = NB; ibo = M;
    } else {
        //  Last index in B is an outer index
        ibi = K; ibo = NB;
    }
    if(idxc1[NC - 1] < N) {
        //  Last index in C comes from A
        ica = NC; icb = M;
    } else {
        //  Last index in C comes from B
        ica = N; icb = NC;
    }

    for(size_t i = 0; i < NA; i++) {
        if(idxa1[NA - i - 1] >= NC) {
            idxa2[iai - 1] = idxa1[NA - i - 1];
            iai--;
        } else {
            idxa2[iao - 1] = idxa1[NA - i - 1];
            iao--;
        }
    }
    for(size_t i = 0; i < NB; i++) {
        if(idxb1[NB - i - 1] >= NC) {
            idxb2[ibi - 1] = idxb1[NB - i - 1];
            ibi--;
        } else {
            idxb2[ibo - 1] = idxb1[NB - i - 1];
            ibo--;
        }
    }
    for(size_t i = 0; i < NC; i++) {
        if(idxc1[NC - i - 1] < N) {
            idxc2[ica - 1] = idxc1[NC - i - 1];
            ica--;
        } else {
            idxc2[icb - 1] = idxc1[NC - i - 1];
            icb--;
        }
    }

    bool lasta_i = (idxa2[NA - 1] >= NC);
    bool lastb_i = (idxb2[NB - 1] >= NC);
    bool lastc_a = (idxc2[NC - 1] < N);

    if(lastc_a) {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ji) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ji) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ji) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxa2[K + i] = idxc2[M + i];
                for(size_t i = 0; i < M; i++) idxc2[i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    } else {
        if(lasta_i) {
            if(lastb_i) {
                //  C(ij) = A(ik) B(jk)
                for(size_t i = 0; i < N; i++) idxa2[i] = idxc2[i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[N + i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ik) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[i];
                for(size_t i = 0; i < M; i++) idxb2[K + i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[N + i];
            }
        } else {
            if(lastb_i) {
                //  C(ij) = A(ki) B(jk)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxb2[i] = idxc2[N + i];
                for(size_t i = 0; i < K; i++) idxa2[i] = idxb2[M + i];
            } else {
                //  C(ij) = A(ki) B(kj)
                for(size_t i = 0; i < N; i++) idxc2[i] = idxa2[K + i];
                for(size_t i = 0; i < M; i++) idxc2[N + i] = idxb2[K + i];
                for(size_t i = 0; i < K; i++) idxb2[i] = idxa2[i];
            }
        }
    }

    permutation_builder<NA> pba(idxa2, idxa1);
    permutation_builder<NB> pbb(idxb2, idxb1);
    permutation_builder<NC> pbc(idxc2, idxc1);
    m_perma.permute(pba.get_perm());
    m_permb.permute(pbb.get_perm());
    m_permc.permute(pbc.get_perm());

}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_ALIGN_H_
