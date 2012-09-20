#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_CLST_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_CLST_IMPL_H

#include <set>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "gen_bto_contract2_clst.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
const char *gen_bto_contract2_clst<N, M, K, Traits>::k_clazz =
    "gen_bto_contract2_clst<N, M, K, T>";


template<size_t N, size_t M, typename Traits>
const char *gen_bto_contract2_clst<N, M, 0, Traits>::k_clazz =
    "gen_bto_contract2_clst<N, M, 0, T>";


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_clst<N, M, K, Traits>::gen_bto_contract2_clst(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const orbit_list<NA, element_type> &ola,
    const orbit_list<NB, element_type> &olb,
    const dimensions<NA> &bidimsa,
    const dimensions<NB> &bidimsb,
    const dimensions<NC> &bidimsc,
    const index<NC> &ic) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_ola(ola), m_olb(olb),
    m_bidimsa(bidimsa), m_bidimsb(bidimsb), m_bidimsc(bidimsc), m_ic(ic) {

}


template<size_t N, size_t M, typename Traits>
gen_bto_contract2_clst<N, M, 0, Traits>::gen_bto_contract2_clst(
    const contraction2<N, M, 0> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const orbit_list<NA, element_type> &ola,
    const orbit_list<NB, element_type> &olb,
    const dimensions<NA> &bidimsa,
    const dimensions<NB> &bidimsb,
    const dimensions<NC> &bidimsc,
    const index<NC> &ic) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_ola(ola), m_olb(olb),
    m_bidimsa(bidimsa), m_bidimsb(bidimsb), m_bidimsc(bidimsc), m_ic(ic) {

}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst<N, M, K, Traits>::build_list(bool testzero) {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.

    //  (The abbreviated version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

    const sequence<NA + NB + NC, size_t> &conn = m_contr.get_conn();
    const symmetry<NA, element_type> &syma = ca.req_const_symmetry();
    const symmetry<NB, element_type> &symb = cb.req_const_symmetry();

    index<K> ik1, ik2;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] > NC) {
            ik2[j++] = m_bidimsa[i] - 1;
        }
    }
    dimensions<K> bidimsk(index_range<K>(ik1, ik2));
    std::set<size_t> ikset;
    size_t nk = bidimsk.get_size();
    for(size_t i = 0; i < nk; i++) ikset.insert(i);

    while(!ikset.empty()) {

        index<NA> ia;
        index<NB> ib;
        const index<NC> &ic = m_ic;
        index<K> ik;
        abs_index<K>::get_index(*ikset.begin(), bidimsk, ik);
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

        orbit<NA, element_type> oa(syma, ia, false);
        orbit<NB, element_type> ob(symb, ib, false);

        bool zero = !m_ola.contains(oa.get_acindex()) ||
            !m_olb.contains(ob.get_acindex());

        abs_index<NA> acia(oa.get_acindex(), m_bidimsa);
        abs_index<NB> acib(ob.get_acindex(), m_bidimsb);
        if(!zero) {
            zero = ca.req_is_zero_block(acia.get_index()) ||
                cb.req_is_zero_block(acib.get_index());
        }

        contr_list clst;

        //  Build the list of contractions for the current orbits A, B

        typename orbit<NA, element_type>::iterator ja;
        typename orbit<NB, element_type>::iterator jb;
        for(ja = oa.begin(); ja != oa.end(); ++ja)
        for(jb = ob.begin(); jb != ob.end(); ++jb) {
            index<NA> ia1;
            index<NB> ib1;
            abs_index<NA>::get_index(oa.get_abs_index(ja), m_bidimsa, ia1);
            abs_index<NB>::get_index(ob.get_abs_index(jb), m_bidimsb, ib1);
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
            if(!zero) {
                clst.push_back(contr_pair(oa.get_acindex(), ob.get_acindex(),
                    oa.get_transf(ja), ob.get_transf(jb)));
            }
            ikset.erase(abs_index<K>::get_abs_index(ika, bidimsk));
        }

        coalesce(clst);
        bool clst_empty = clst.empty();
        merge(clst); // This empties clst

        //  In the abbreviated version of the algorithm, if the list is
        //  not empty, there is no need to continue: the block is non-zero

        if(testzero && !clst_empty) break;
    }
}


template<size_t N, size_t M, typename Traits>
void gen_bto_contract2_clst<N, M, 0, Traits>::build_list(bool testzero) {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.

    //  (The abbreviated version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);

    const sequence<NA + NB + NC, size_t> &conn = m_contr.get_conn();
    const symmetry<NA, element_type> &syma = ca.req_const_symmetry();
    const symmetry<NB, element_type> &symb = cb.req_const_symmetry();

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

    orbit<NA, element_type> oa(syma, ia, false);
    orbit<NB, element_type> ob(symb, ib, false);

    bool zero = !m_ola.contains(oa.get_acindex()) ||
        !m_olb.contains(ob.get_acindex());

    abs_index<N> acia(oa.get_acindex(), m_bidimsa);
    abs_index<M> acib(ob.get_acindex(), m_bidimsb);
    if(!zero) {
        zero = ca.req_is_zero_block(acia.get_index()) ||
            cb.req_is_zero_block(acib.get_index());
    }

    contr_list clst;

    //  Build the list of contractions for the current orbits A, B

    typename orbit<NA, element_type>::iterator ja;
    typename orbit<NB, element_type>::iterator jb;
    for(ja = oa.begin(); ja != oa.end(); ++ja)
    for(jb = ob.begin(); jb != ob.end(); ++jb) {
        index<NA> ia1;
        index<NB> ib1;
        abs_index<NA>::get_index(oa.get_abs_index(ja), m_bidimsa, ia1);
        abs_index<NB>::get_index(ob.get_abs_index(jb), m_bidimsb, ib1);
        index<NC> ic1;
        for(size_t i = 0; i < NC; i++) {
            if(conn[i] >= 2 * N + M) {
                ic1[i] = ib1[conn[i] - 2 * N - M];
            } else {
                ic1[i] = ia1[conn[i] - N - M];
            }
        }
        if(!ic1.equals(ic)) continue;
        if(!zero) {
            clst.push_back(contr_pair(oa.get_acindex(), ob.get_acindex(),
                oa.get_transf(ja), ob.get_transf(jb)));
        }
    }

    coalesce(clst);
    merge(clst);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_base<N, M, K, Traits>::coalesce(contr_list &clst) {

    typename contr_list::iterator j1 = clst.begin();
    while(j1 != clst.end()) {
        typename contr_list::iterator j2 = j1;
        ++j2;
        bool incj1 = true;
        while(j2 != clst.end()) {
            if(j1->tra.get_perm().equals(j2->tra.get_perm()) &&
                j1->trb.get_perm().equals(j2->trb.get_perm())) {
                // TODO: replace with scalar_transf::combine
                // TODO: this code is double-specific!
                double d1 = j1->tra.get_scalar_tr().get_coeff() *
                    j1->trb.get_scalar_tr().get_coeff();
                double d2 = j2->tra.get_scalar_tr().get_coeff() *
                    j2->trb.get_scalar_tr().get_coeff();
                if(d1 + d2 == 0) {
                    j1 = clst.erase(j1);
                    if(j1 == j2) {
                        j1 = j2 = clst.erase(j2);
                    } else {
                        j2 = clst.erase(j2);
                    }
                    incj1 = false;
                    break;
                } else {
                    j1->tra.get_scalar_tr().reset();
                    j1->trb.get_scalar_tr().reset();
                    j1->tra.get_scalar_tr().scale(d1 + d2);
                    j2 = clst.erase(j2);
                }
            } else {
                ++j2;
            }
        }
        if(incj1) ++j1;
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_clst_base<N, M, K, Traits>::merge(contr_list &clst) {

    m_clst.splice(m_clst.end(), clst);
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_CLST_IMPL_H
