#ifndef LIBTENSOR_GEN_BTO_DIRSUM_BIS_IMPL_H
#define LIBTENSOR_GEN_BTO_DIRSUM_BIS_IMPL_H

#include "gen_bto_dirsum_bis.h"

namespace libtensor {


template<size_t N, size_t M>
gen_bto_dirsum_bis<N, M>::gen_bto_dirsum_bis(
        const block_index_space<N> &bisa,
        const block_index_space<M> &bisb,
        const permutation<N + M> &permc) :
        m_bisc(mk_dims(bisa, bisb)) {

    mask<N> mska, mska1;
    mask<M> mskb, mskb1;
    mask<N + M> mskc;
    bool done;
    size_t i;

    i = 0;
    done = false;
    while(!done) {
        while(i < N && mska[i]) i++;
        if(i == N) {
            done = true;
            continue;
        }

        size_t typ = bisa.get_type(i);
        for(size_t j = 0; j < N; j++) {
            mskc[j] = mska1[j] = bisa.get_type(j) == typ;
        }
        const split_points &pts = bisa.get_splits(typ);
        for(size_t j = 0; j < pts.get_num_points(); j++)
            m_bisc.split(mskc, pts[j]);

        mska |= mska1;
    }
    for(size_t j = 0; j < N; j++) mskc[j] = false;

    i = 0;
    done = false;
    while(!done) {
        while(i < M && mskb[i]) i++;
        if(i == M) {
            done = true;
            continue;
        }

        size_t typ = bisb.get_type(i);
        for(size_t j = 0; j < M; j++) {
            mskc[N + j] = mskb1[j] =
                bisb.get_type(j) == typ;
        }
        const split_points &pts = bisb.get_splits(typ);
        for(size_t j = 0; j < pts.get_num_points(); j++)
            m_bisc.split(mskc, pts[j]);

        mskb |= mskb1;
    }

    m_bisc.match_splits();
    m_bisc.permute(permc);
}


template<size_t N, size_t M>
dimensions<N + M> gen_bto_dirsum_bis<N, M>::mk_dims(
        const block_index_space<N> &bisa,
        const block_index_space<M> &bisb) {

    const dimensions<N> &dimsa = bisa.get_dims();
    const dimensions<M> &dimsb = bisb.get_dims();

    index<N + M> i1, i2;
    for(register size_t i = 0; i < N; i++)
        i2[i] = dimsa[i] - 1;
    for(register size_t i = 0; i < M; i++)
        i2[N + i] = dimsb[i] - 1;

    return dimensions<N + M>(index_range<N + M>(i1, i2));
}


} // namespace libtensor


#endif // LIBTENOSR_GEN_BTO_DIRSUM_BIS_IMPL_H
