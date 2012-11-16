#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_LIST_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_LIST_H

#include <algorithm>
#include <vector>
#include <libtensor/core/contraction2.h>
#include <libtensor/core/index.h>
#include "block_list.h"

namespace libtensor {


template<size_t N, size_t M, size_t K>
class gen_bto_contract2_block_list {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

private:
    std::vector< index<2> > m_blsta_1;
    std::vector< index<2> > m_blstb_1;
    std::vector< index<2> > m_blsta_2;
    std::vector< index<2> > m_blstb_2;

public:
    gen_bto_contract2_block_list(
        const contraction2<N, M, K> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb) {

        build_list(contr, bidimsa, blsta, bidimsb, blstb);
    }

    const std::vector< index<2> > &get_blsta_1() const {
        return m_blsta_1;
    }

    const std::vector< index<2> > &get_blstb_1() const {
        return m_blstb_1;
    }

    const std::vector< index<2> > &get_blsta_2() const {
        return m_blsta_2;
    }

    const std::vector< index<2> > &get_blstb_2() const {
        return m_blstb_2;
    }

private:
    void build_list(
        const contraction2<N, M, K> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb);

};


template<size_t N, size_t M>
class gen_bto_contract2_block_list<N, M, 0> {
public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

private:
    std::vector< index<2> > m_blsta_1;
    std::vector< index<2> > m_blstb_1;
    std::vector< index<2> > m_blsta_2;
    std::vector< index<2> > m_blstb_2;

public:
    gen_bto_contract2_block_list(
        const contraction2<N, M, 0> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb) {

        build_list(contr, bidimsa, blsta, bidimsb, blstb);
    }

    const std::vector< index<2> > &get_blsta_1() const {
        return m_blsta_1;
    }

    const std::vector< index<2> > &get_blstb_1() const {
        return m_blstb_1;
    }

    const std::vector< index<2> > &get_blsta_2() const {
        return m_blsta_2;
    }

    const std::vector< index<2> > &get_blstb_2() const {
        return m_blstb_2;
    }

private:
    void build_list(
        const contraction2<N, M, 0> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb) { }

};


struct gen_bto_contract2_block_list_less_1 {
    bool operator()(
        const index<2> &i1,
        const index<2> &i2) {

        if(i1[0] < i2[0]) return true;
        if(i1[0] == i2[0]) return i1[1] < i2[1];
        return false;
    }
};


struct gen_bto_contract2_block_list_less_2 {
    bool operator()(
        const index<2> &i1,
        const index<2> &i2) {

        if(i1[1] < i2[1]) return true;
        if(i1[1] == i2[1]) return i1[0] < i2[0];
        return false;
    }
};


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_block_list<N, M, K>::build_list(
    const contraction2<N, M, K> &contr,
    const dimensions<NA> &bidimsa,
    const block_list<NA> &blsta,
    const dimensions<NB> &bidimsb,
    const block_list<NB> &blstb) {

    const sequence<2 * (N + M + K), size_t> &conn = contr.get_conn();

    sequence<N, size_t> mapai;
    sequence<M, size_t> mapbj;
    sequence<K, size_t> mapak, mapbk;
    index<N> ii1, ii2;
    index<M> ij1, ij2;
    index<K> ik1, ik2;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] < NC) {
            mapai[j] = i;
            ii2[j] = bidimsa[i] - 1;
            j++;
        }
    }
    for(size_t i = 0, j = 0; i < NB; i++) {
        if(conn[NC + NA + i] < NC) {
            mapbj[j] = i;
            ij2[j] = bidimsb[i] - 1;
            j++;
        }
    }
    for(size_t i = 0, k = 0; i < NA; i++) {
        if(conn[NC + i] >= NC + NA) {
            mapak[k] = i;
            mapbk[k] = conn[NC + i] - NC - NA;
            ik2[k] = bidimsa[i] - 1;
            k++;
        }
    }

    dimensions<N> dimsi(index_range<N>(ii1, ii2));
    dimensions<M> dimsj(index_range<M>(ij1, ij2));
    dimensions<K> dimsk(index_range<K>(ik1, ik2));

    index<NA> bidxa;
    index<NB> bidxb;
    index<N> idxi;
    index<M> idxj;
    index<K> idxk;
    index<2> idx2;

    for(typename block_list<NA>::iterator ia = blsta.begin();
        ia != blsta.end(); ++ia) {

        abs_index<NA>::get_index(blsta.get_abs_index(ia), bidimsa, bidxa);
        for(size_t i = 0; i < N; i++) idxi[i] = bidxa[mapai[i]];
        for(size_t i = 0; i < K; i++) idxk[i] = bidxa[mapak[i]];
        idx2[0] = abs_index<K>::get_abs_index(idxk, dimsk);
        idx2[1] = abs_index<N>::get_abs_index(idxi, dimsi);
        m_blsta_1.push_back(idx2);
    }

    for(typename block_list<NB>::iterator ib = blstb.begin();
        ib != blstb.end(); ++ib) {

        abs_index<NB>::get_index(blstb.get_abs_index(ib), bidimsb, bidxb);
        for(size_t i = 0; i < M; i++) idxj[i] = bidxb[mapbj[i]];
        for(size_t i = 0; i < K; i++) idxk[i] = bidxb[mapbk[i]];
        idx2[0] = abs_index<K>::get_abs_index(idxk, dimsk);
        idx2[1] = abs_index<M>::get_abs_index(idxj, dimsj);
        m_blstb_1.push_back(idx2);
    }

    m_blsta_2 = m_blsta_1;
    m_blstb_2 = m_blstb_1;

    gen_bto_contract2_block_list_less_1 comp_1;
    std::sort(m_blsta_1.begin(), m_blsta_1.end(), comp_1);
    std::sort(m_blstb_1.begin(), m_blstb_1.end(), comp_1);
    gen_bto_contract2_block_list_less_2 comp_2;
    std::sort(m_blsta_2.begin(), m_blsta_2.end(), comp_2);
    std::sort(m_blstb_2.begin(), m_blstb_2.end(), comp_2);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_LIST_H
