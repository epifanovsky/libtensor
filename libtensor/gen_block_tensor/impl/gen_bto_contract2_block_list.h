#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_LIST_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_BLOCK_LIST_H

#include <algorithm>
#include <utility>
#include <vector>
#include <libtensor/core/contraction2.h>
#include "block_list.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_block_list {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

private:
    std::vector< std::pair<size_t, size_t> > m_blsta;
    std::vector< std::pair<size_t, size_t> > m_blstb;

public:
    gen_bto_contract2_block_list(
        const contraction2<N, M, K> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb) {

        build_list(contr, bidimsa, blsta, bidimsb, blstb);
    }

    const std::vector< std::pair<size_t, size_t> > &get_blsta() const {
        return m_blsta;
    }

    const std::vector< std::pair<size_t, size_t> > &get_blstb() const {
        return m_blstb;
    }

private:
    void build_list(
        const contraction2<N, M, K> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb);

};


template<size_t N, size_t M, typename Traits>
class gen_bto_contract2_block_list<N, M, 0, Traits> {
public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

private:
    std::vector< std::pair<size_t, size_t> > m_blsta_1;
    std::vector< std::pair<size_t, size_t> > m_blstb_1;
    std::vector< std::pair<size_t, size_t> > m_blsta_2;
    std::vector< std::pair<size_t, size_t> > m_blstb_2;

public:
    gen_bto_contract2_block_list(
        const contraction2<N, M, 0> &contr,
        const dimensions<NA> &bidimsa,
        const block_list<NA> &blsta,
        const dimensions<NB> &bidimsb,
        const block_list<NB> &blstb) {

        build_list(contr, bidimsa, blsta, bidimsb, blstb);
    }

    const std::vector< std::pair<size_t, size_t> > &get_blsta_1() const {
        return m_blsta_1;
    }

    const std::vector< std::pair<size_t, size_t> > &get_blstb_1() const {
        return m_blstb_1;
    }

    const std::vector< std::pair<size_t, size_t> > &get_blsta_2() const {
        return m_blsta_2;
    }

    const std::vector< std::pair<size_t, size_t> > &get_blstb_2() const {
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
        const std::pair<size_t, size_t> &a,
        const std::pair<size_t, size_t> &b) {

        if(a.first < b.first) return true;
        if(a.first == b.first) return a.second < b.second;
        return false;
    }
};


struct gen_bto_contract2_block_list_less_2 {
    bool operator()(
        const std::pair<size_t, size_t> &a,
        const std::pair<size_t, size_t> &b) {

        if(a.second < b.second) return true;
        if(a.second == b.second) return a.first < b.first;
        return false;
    }
};


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_block_list<N, M, K, Traits>::build_list(
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

    for(typename block_list<NA>::iterator ia = blsta.begin();
        ia != blsta.end(); ++ia) {

        abs_index<NA>::get_index(blsta.get_abs_index(ia), bidimsa, bidxa);
        for(size_t i = 0; i < N; i++) idxi[i] = bidxa[mapai[i]];
        for(size_t i = 0; i < K; i++) idxk[i] = bidxa[mapak[i]];
        m_blsta.push_back(std::make_pair(
            abs_index<K>::get_abs_index(idxk, dimsk),
            abs_index<N>::get_abs_index(idxi, dimsi)));
    }

    for(typename block_list<NB>::iterator ib = blstb.begin();
        ib != blstb.end(); ++ib) {

        abs_index<NB>::get_index(blstb.get_abs_index(ib), bidimsb, bidxb);
        for(size_t i = 0; i < M; i++) idxj[i] = bidxb[mapbj[i]];
        for(size_t i = 0; i < K; i++) idxk[i] = bidxb[mapbk[i]];
        m_blstb.push_back(std::make_pair(
            abs_index<K>::get_abs_index(idxk, dimsk),
            abs_index<M>::get_abs_index(idxj, dimsj)));
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
