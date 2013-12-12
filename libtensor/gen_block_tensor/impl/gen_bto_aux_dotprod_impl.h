#ifndef LIBTENSOR_GEN_BTO_AUX_DOTPROD_IMPL_H
#define LIBTENSOR_GEN_BTO_AUX_DOTPROD_IMPL_H

#include <libutil/threads/auto_lock.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include "../gen_bto_aux_dotprod.h"

namespace libtensor {


template<size_t N, typename Traits>
const char gen_bto_aux_dotprod<N, Traits>::k_clazz[] =
    "gen_bto_aux_dotprod<N, Traits>";


template<size_t N, typename Traits>
gen_bto_aux_dotprod<N, Traits>::gen_bto_aux_dotprod(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf_type &tra,
    const symmetry<N, element_type> &symb) :

    m_bta(bta), m_tra(tra), m_bisb(symb.get_bis()), m_symb(symb.get_bis()),
    m_symc(symb.get_bis()), m_d(Traits::zero()) {

    so_copy<N, element_type>(symb).perform(m_symb);

    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);
    const symmetry<N, element_type> &syma = ca.req_const_symmetry();

    sequence<N, size_t> seq1a, seq2a;
    for(size_t ii = 0; ii < N; ii++) {
        seq1a[ii] = ii;
        seq2a[ii] = N + ii;
    }
    tra.get_perm().apply(seq1a);
    sequence<N + N, size_t> seq1b, seq2b;
    for(size_t ii = 0; ii < N; ii++) {
        seq1b[ii] = ii;
        seq1b[N + ii] = N + ii;
        seq2b[ii] = seq1a[ii];
        seq2b[N + ii] = seq2a[ii];
    }
    permutation_builder<N + N> pbb(seq2b, seq1b);

    block_index_space_product_builder<N, N> bbx(bta.get_bis(), m_bisb,
        pbb.get_perm());

    symmetry<N + N, element_type> symx(bbx.get_bis());
    so_dirprod<N, N, element_type>(syma, m_symb, pbb.get_perm()).perform(symx);

    mask<N + N> msk;
    sequence<N + N, size_t> seq;
    for(size_t ii = 0; ii < N; ii++) {
        msk[ii] = msk[ii + N] = true;
        seq[ii] = seq[ii + N] = ii;
    }
    so_merge<N + N, N, element_type>(symx, msk, seq).perform(m_symc);
}


template<size_t N, typename Traits>
gen_bto_aux_dotprod<N, Traits>::~gen_bto_aux_dotprod() {

}


template<size_t N, typename Traits>
void gen_bto_aux_dotprod<N, Traits>::open() {

    m_d = Traits::zero();
}


template<size_t N, typename Traits>
void gen_bto_aux_dotprod<N, Traits>::close() {

}


template<size_t N, typename Traits>
void gen_bto_aux_dotprod<N, Traits>::put(
    const index<N> &idxb,
    rd_block_type &blkb,
    const tensor_transf<N, element_type> &tr) {

    typedef typename Traits::template to_dotprod_type<N>::type to_dotprod_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);
    const symmetry<N, element_type> &syma = ca.req_const_symmetry();

    permutation<N> pinva(m_tra.get_perm(), true);

    orbit<N, element_type> ob(m_symb, idxb);

    dimensions<N> bidimsb = m_bisb.get_block_index_dims();
    size_t aidxb = abs_index<N>::get_abs_index(idxb, bidimsb);
    subgroup_orbits<N, element_type> sgo(m_symb, m_symc, aidxb);
    for(typename subgroup_orbits<N, element_type>::iterator i = sgo.begin();
        i != sgo.end(); ++i) {

        index<N> idxc;
        sgo.get_index(i, idxc);

        orbit<N, element_type> oc(m_symc, idxc);
        scalar_transf_sum<element_type> sum;
        for(typename orbit<N, element_type>::iterator ioc = oc.begin();
            ioc != oc.end(); ++ioc) {
            sum.add(oc.get_transf(ioc).get_scalar_tr());
        }
        if(sum.is_zero()) continue;

        index<N> idxa(idxc);
        idxa.permute(pinva);

        orbit<N, element_type> oa(syma, idxa, true);
        if(!oa.is_allowed() || ca.req_is_zero_block(oa.get_cindex())) continue;

        tensor_transf<N, element_type> tra(oa.get_transf(idxa)), trb(tr);
        tra.transform(m_tra);
        trb.transform(ob.get_transf(idxc));

        rd_block_type &blka = ca.req_const_block(oa.get_cindex());
        element_type d = to_dotprod_type(blka, tra, blkb, trb).calculate();
        ca.ret_const_block(oa.get_cindex());

        sum.apply(d);
        {
            libutil::auto_lock<libutil::mutex> lock(m_mtx);
            m_d += d;
        }
    }
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_AUX_DOTPROD_IMPL_H
