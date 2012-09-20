#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BIS_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BIS_IMPL_H

#include "gen_bto_contract2_bis.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
gen_bto_contract2_bis<N, M, K>::gen_bto_contract2_bis(
    const contraction2<N, M, K> &contr,
    const block_index_space<N + K> &bisa,
    const block_index_space<M + K> &bisb) :

    m_dimsc(contr, bisa.get_dims(), bisb.get_dims()),
    m_bisc(m_dimsc.get_dimsc()) {

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();
    const dimensions<N + K> &dimsa = bisa.get_dims();
    const dimensions<M + K> &dimsb = bisb.get_dims();

    mask<N + K> mdonea;
    mask<M + K> mdoneb;

    for(size_t ia = 0; ia < N + K; ia++) if(!mdonea[ia]) {
        size_t typ = bisa.get_type(ia);
        mask<N + K> ma;
        mask<N + M> mc;
        for(size_t i = ia; i < N + K; i++) {
            ma[i] = (bisa.get_type(i) == typ);
            size_t ic = conn[N + M + i];
            if(ic < N + M) mc[ic] = ma[i];
        }
        const split_points &pts = bisa.get_splits(typ);
        for(size_t ipt = 0; ipt < pts.get_num_points(); ipt++) {
            m_bisc.split(mc, pts[ipt]);
        }
        mdonea |= ma;
    }
    for(size_t ib = 0; ib < M + K; ib++) if(!mdoneb[ib]) {
        size_t typ = bisb.get_type(ib);
        mask<M + K> mb;
        mask<N + M> mc;
        for(size_t i = ib; i < M + K; i++) {
            mb[i] = (bisb.get_type(i) == typ);
            size_t ic = conn[N + M + N + K + i];
            if(ic < N + M) mc[ic] = mb[i];
        }
        const split_points &pts = bisb.get_splits(typ);
        for(size_t ipt = 0; ipt < pts.get_num_points(); ipt++) {
            m_bisc.split(mc, pts[ipt]);
        }
        mdoneb |= mb;
    }
    m_bisc.match_splits();
}


template<size_t N, size_t M>
gen_bto_contract2_bis<N, M, 0>::gen_bto_contract2_bis(
    const contraction2<N, M, 0> &contr,
    const block_index_space<N> &bisa,
    const block_index_space<M> &bisb) :

    m_dimsc(contr, bisa.get_dims(), bisb.get_dims()),
    m_bisc(m_dimsc.get_dimsc()) {

    const sequence<2 * (N + M), size_t> &conn = contr.get_conn();
    const dimensions<N> &dimsa = bisa.get_dims();
    const dimensions<M> &dimsb = bisb.get_dims();

    mask<N> mdonea;
    mask<M> mdoneb;

    for(size_t ia = 0; ia < N; ia++) if(!mdonea[ia]) {
        size_t typ = bisa.get_type(ia);
        mask<N> ma;
        mask<N + M> mc;
        for(size_t i = ia; i < N; i++) {
            ma[i] = (bisa.get_type(i) == typ);
            size_t ic = conn[N + M + i];
            if(ic < N + M) mc[ic] = ma[i];
        }
        const split_points &pts = bisa.get_splits(typ);
        for(size_t ipt = 0; ipt < pts.get_num_points(); ipt++) {
            m_bisc.split(mc, pts[ipt]);
        }
        mdonea |= ma;
    }
    for(size_t ib = 0; ib < M; ib++) if(!mdoneb[ib]) {
        size_t typ = bisb.get_type(ib);
        mask<M> mb;
        mask<N + M> mc;
        for(size_t i = ib; i < M; i++) {
            mb[i] = (bisb.get_type(i) == typ);
            size_t ic = conn[N + M + N + i];
            if(ic < N + M) mc[ic] = mb[i];
        }
        const split_points &pts = bisb.get_splits(typ);
        for(size_t ipt = 0; ipt < pts.get_num_points(); ipt++) {
            m_bisc.split(mc, pts[ipt]);
        }
        mdoneb |= mb;
    }
    m_bisc.match_splits();
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BIS_IMPL_H
