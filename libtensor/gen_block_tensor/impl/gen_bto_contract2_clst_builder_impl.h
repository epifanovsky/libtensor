#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_IMPL_H

#include <cstring>
#include <libutil/threads/tls.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "gen_bto_contract2_clst_builder.h"

namespace libtensor {


class gen_bto_contract2_clst_builder_buffer {
private:
    std::vector<char> m_v;

public:
    static std::vector<char> &get_v() {
        return libutil::tls<gen_bto_contract2_clst_builder_buffer>::
            get_instance().get().m_v;
    }

};


template<size_t N, size_t M, size_t K, typename Traits>
const char *gen_bto_contract2_clst_builder<N, M, K, Traits>::k_clazz =
    "gen_bto_contract2_clst_builder<N, M, K, Traits>";


template<size_t N, size_t M, typename Traits>
const char *gen_bto_contract2_clst_builder<N, M, 0, Traits>::k_clazz =
    "gen_bto_contract2_clst_builder<N, M, 0, Traits>";


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_clst_builder<N, M, K, Traits>::gen_bto_contract2_clst_builder(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    const block_list<NA> &blka,
    const block_list<NB> &blkb,
    const dimensions<NC> &bidimsc,
    const index<NC> &ic) :

    gen_bto_contract2_clst_builder_base<N, M, K, Traits>(contr),
    m_syma(syma), m_symb(symb), m_blka(blka), m_blkb(blkb), m_bidimsc(bidimsc),
    m_ic(ic) {

}


template<size_t N, size_t M, typename Traits>
gen_bto_contract2_clst_builder<N, M, 0, Traits>::gen_bto_contract2_clst_builder(
    const contraction2<N, M, 0> &contr,
    const symmetry<NA, element_type> &syma,
    const symmetry<NB, element_type> &symb,
    const block_list<NA> &blka,
    const block_list<NB> &blkb,
    const dimensions<NC> &bidimsc,
    const index<NC> &ic) :

    gen_bto_contract2_clst_builder_base<N, M, 0, Traits>(contr),
    m_syma(syma), m_symb(symb), m_blka(blka), m_blkb(blkb), m_bidimsc(bidimsc),
    m_ic(ic) {

}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_builder<N, M, K, Traits>::build_list(
    bool testzero) {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.

    //  (The abbreviated version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    const sequence<NA + NB + NC, size_t> &conn = get_contr().get_conn();
    const dimensions<NA> &bidimsa = m_blka.get_dims();
    const dimensions<NB> &bidimsb = m_blkb.get_dims();

    index<K> ik1, ik2;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] > NC) {
            ik2[j++] = bidimsa[i] - 1;
        }
    }
    dimensions<K> bidimsk(index_range<K>(ik1, ik2));
    size_t nk = bidimsk.get_size();
    std::vector<char> &chk = gen_bto_contract2_clst_builder_buffer::get_v();
    chk.resize(nk, 0);
    ::memset(&chk[0], 1, nk);

    size_t aik = 0;
    const char *p0 = &chk[0];
    while(aik < nk) {

        const char *p = (const char*)::memchr(p0 + aik, 1, nk - aik);
        if(p == 0) break;
        aik = p - p0;

        index<NA> ia;
        index<NB> ib;
        const index<NC> &ic = m_ic;
        index<K> ik;
        abs_index<K>::get_index(aik, bidimsk, ik);
        sequence<K, size_t> ka(0), kb(0);

        //  Determine ia, ib from ic, ik
        for(size_t i = 0, j = 0; i < NA; i++) {
            if(conn[NC + i] < NC) {
                ia[i] = ic[conn[NC + i]];
            } else {
                ka[j] = i;
                kb[j] = conn[NC + i] - 2 * N - M - K;
                ia[ka[j]] = ib[kb[j]] = ik[j];
                j++;
            }
        }
        for(size_t i = 0; i < NB; i++) {
            if(conn[2 * N + M + K + i] < N + M) {
                ib[i] = ic[conn[2 * N + M + K + i]];
            }
        }

        size_t aia = abs_index<NA>::get_abs_index(ia, bidimsa);
        size_t aib = abs_index<NB>::get_abs_index(ib, bidimsb);
        if(!m_blka.contains(aia) || !m_blkb.contains(aib)) {
            chk[aik] = 0;
            continue;
        }

        orbit<NA, element_type> oa(m_syma, ia, false);
        orbit<NB, element_type> ob(m_symb, ib, false);

        contr_list clst;

        //  Build the list of contractions for the current orbits A, B

        typename orbit<NA, element_type>::iterator ja;
        typename orbit<NB, element_type>::iterator jb;
        for(ja = oa.begin(); ja != oa.end(); ++ja)
        for(jb = ob.begin(); jb != ob.end(); ++jb) {
            index<NA> ia1;
            index<NB> ib1;
            abs_index<NA>::get_index(oa.get_abs_index(ja), bidimsa, ia1);
            abs_index<NB>::get_index(ob.get_abs_index(jb), bidimsb, ib1);
            index<NC> ic1;
            index<K> ika, ikb;
            for(size_t i = 0; i < K; i++) {
                ika[i] = ia1[ka[i]];
                ikb[i] = ib1[kb[i]];
            }
            if(!ika.equals(ikb)) continue;
            for(size_t i = 0; i < N + M; i++) {
                if(conn[i] >= 2 * N + M + K) {
                    ic1[i] = ib1[conn[i] - 2 * N - M - K];
                } else {
                    ic1[i] = ia1[conn[i] - N - M];
                }
            }
            if(!ic1.equals(ic)) continue;
            clst.push_back(contr_pair(
                oa.get_abs_index(ja), oa.get_acindex(), oa.get_transf(ja),
                ob.get_abs_index(jb), ob.get_acindex(), ob.get_transf(jb)));
            chk[abs_index<K>::get_abs_index(ika, bidimsk)] = 0;
        }

        coalesce(clst);
        bool clst_empty = clst.empty();
        merge(clst); // This empties clst

        //  In the abbreviated version of the algorithm, if the list is
        //  not empty, there is no need to continue: the block is non-zero

        if(testzero && !clst_empty) break;
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_builder<N, M, K, Traits>::build_list(
    bool testzero, const gen_bto_contract2_block_list<N, M, K> &bl) {

    if(testzero == true) {
        build_list(true);
        return;
    }

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.

    //  (The abbreviated version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    const sequence<NA + NB + NC, size_t> &conn = get_contr().get_conn();
    const dimensions<NA> &bidimsa = m_blka.get_dims();
    const dimensions<NB> &bidimsb = m_blkb.get_dims();

    const index<NC> &ic = m_ic;
    sequence<N, size_t> mapai;
    sequence<M, size_t> mapbj;
    sequence<K, size_t> mapak, mapbk;
    index<N> ii1, ii2, ii;
    index<M> ij1, ij2, ij;
    index<K> ik1, ik2, ik;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] < NC) {
            mapai[j] = i;
            ii2[j] = bidimsa[i] - 1;
            ii[j] = ic[conn[NC + i]];
            j++;
        }
    }
    for(size_t i = 0, j = 0; i < NB; i++) {
        if(conn[NC + NA + i] < NC) {
            mapbj[j] = i;
            ij2[j] = bidimsb[i] - 1;
            ij[j] = ic[conn[NC + NA + i]];
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
    size_t nk = dimsk.get_size();

    size_t aii = abs_index<N>::get_abs_index(ii, dimsi);
    size_t aij = abs_index<M>::get_abs_index(ij, dimsj);
    size_t aik = 0;

    const std::vector< index<2> > &bla = bl.get_blsta_2();
    const std::vector< index<2> > &blb = bl.get_blstb_2();

    index<2> i2;
    i2[0] = aik; i2[1] = aii;
    typename std::vector< index<2> >::const_iterator ibla_beg =
        std::lower_bound(bla.begin(), bla.end(), i2,
            gen_bto_contract2_block_list_less_2());
    i2[0] = aik; i2[1] = aii + 1;
    typename std::vector< index<2> >::const_iterator ibla_end =
        std::lower_bound(ibla_beg, bla.end(), i2,
            gen_bto_contract2_block_list_less_2());
    i2[0] = aik; i2[1] = aij;
    typename std::vector< index<2> >::const_iterator iblb_beg =
        std::lower_bound(blb.begin(), blb.end(), i2,
            gen_bto_contract2_block_list_less_2());
    i2[0] = aik; i2[1] = aij + 1;
    typename std::vector< index<2> >::const_iterator iblb_end =
        std::lower_bound(iblb_beg, blb.end(), i2,
            gen_bto_contract2_block_list_less_2());
    typename std::vector< index<2> >::const_iterator ibla;
    typename std::vector< index<2> >::const_iterator iblb;

    for(ibla = ibla_beg; ibla != ibla_end; ++ibla) {
        index<N> iii;
        index<K> iik;
        abs_index<K>::get_index(ibla->at(0), dimsk, iik);
        if(N>0) abs_index<N>::get_index(ibla->at(1), dimsi, iii);
    }
    for(iblb = iblb_beg; iblb != iblb_end; ++iblb) {
        index<M> iij;
        index<K> iik;
        abs_index<K>::get_index(iblb->at(0), dimsk, iik);
        if(M>0) abs_index<M>::get_index(iblb->at(1), dimsj, iij);
    }

    index<NA> ia;
    index<NB> ib;
    contr_list clst;

    ibla = ibla_beg;
    iblb = iblb_beg;

    while(true) {

        while(ibla != ibla_end && iblb != iblb_end &&
            ibla->at(0) != iblb->at(0)) {

            while(ibla != ibla_end && ibla->at(0) < iblb->at(0)) ++ibla;
            while(iblb != iblb_end && iblb->at(0) < ibla->at(0)) ++iblb;
        }
        if(ibla == ibla_end || iblb == iblb_end) break;

        aik = ibla->at(0);
        abs_index<K>::get_index(aik, dimsk, ik);

        for(size_t i = 0; i < N; i++) ia[mapai[i]] = ii[i];
        for(size_t i = 0; i < M; i++) ib[mapbj[i]] = ij[i];
        for(size_t i = 0; i < K; i++) ia[mapak[i]] = ib[mapbk[i]] = ik[i];

        size_t aia = abs_index<NA>::get_abs_index(ia, bidimsa);
        size_t aib = abs_index<NB>::get_abs_index(ib, bidimsb);
        orbit<NA, element_type> oa(m_syma, ia, false);
        orbit<NB, element_type> ob(m_symb, ib, false);
        clst.push_back(contr_pair(
            aia, oa.get_acindex(), oa.get_transf(aia),
            aib, ob.get_acindex(), ob.get_transf(aib)));

        ++ibla;
        ++iblb;
    }

    coalesce(clst);
    merge(clst); // This empties clst
}


template<size_t N, size_t M, typename Traits>
void gen_bto_contract2_clst_builder<N, M, 0, Traits>::build_list(
    bool testzero) {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.

    //  (The abbreviated version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    const sequence<NA + NB + NC, size_t> &conn = get_contr().get_conn();
    const dimensions<NA> &bidimsa = m_blka.get_dims();
    const dimensions<NB> &bidimsb = m_blkb.get_dims();

    index<N> ia;
    index<M> ib;
    const index<N + M> &ic = m_ic;

    //  Determine ia, ib from ic
    for(size_t i = 0, j = 0; i < N; i++) {
        ia[i] = ic[conn[NC + i]];
    }
    for(size_t i = 0; i < M; i++) {
        ib[i] = ic[conn[NC + NA + i]];
    }

    if(!m_blka.contains(ia) || !m_blkb.contains(ib)) return;

    orbit<NA, element_type> oa(m_syma, ia, false);
    orbit<NB, element_type> ob(m_symb, ib, false);

    contr_list clst;

    //  Build the list of contractions for the current orbits A, B

    typename orbit<NA, element_type>::iterator ja;
    typename orbit<NB, element_type>::iterator jb;
    for(ja = oa.begin(); ja != oa.end(); ++ja)
    for(jb = ob.begin(); jb != ob.end(); ++jb) {
        index<NA> ia1;
        index<NB> ib1;
        abs_index<NA>::get_index(oa.get_abs_index(ja), bidimsa, ia1);
        abs_index<NB>::get_index(ob.get_abs_index(jb), bidimsb, ib1);
        index<NC> ic1;
        for(size_t i = 0; i < NC; i++) {
            if(conn[i] >= 2 * N + M) {
                ic1[i] = ib1[conn[i] - 2 * N - M];
            } else {
                ic1[i] = ia1[conn[i] - N - M];
            }
        }
        if(!ic1.equals(ic)) continue;
        clst.push_back(contr_pair(
            oa.get_abs_index(ja), oa.get_acindex(), oa.get_transf(ja),
            ob.get_abs_index(jb), ob.get_acindex(), ob.get_transf(jb)));
    }

    coalesce(clst);
    merge(clst);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_builder_base<N, M, K, Traits>::coalesce(
    contr_list &clst) {

    typedef typename Traits::template
        to_contract2_type<N, M, K>::clst_optimize_type coalesce_type;

    coalesce_type(m_contr).perform(clst);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_builder_base<N, M, K, Traits>::merge(
    contr_list &clst) {

    m_clst.splice(m_clst.end(), clst);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_BUILDER_IMPL_H
