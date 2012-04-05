#ifndef LIBTENSOR_BTO_APPLY_IMPL_H
#define LIBTENSOR_BTO_APPLY_IMPL_H

#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_apply.h>

namespace libtensor {


template<size_t N, typename Traits>
const char *bto_apply<N, Traits>::k_clazz = "bto_apply<N, Traits>";


template<size_t N, typename Traits>
bto_apply<N, Traits>::bto_apply(block_tensor_t &bta,
        const functor_t &fn, const scalar_tr_t &c) :

    m_bta(bta), m_fn(fn), m_tr(permutation<N>(), c), m_bis(m_bta.get_bis()),
    m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    //! Type of block tensor control object
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    so_apply<N, element_t>(ctrla.req_const_symmetry(), m_tr.get_perm(),
            m_fn.transf(true), m_fn.transf(false),
            m_fn.keep_zero()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Traits>
bto_apply<N, Traits>::bto_apply(block_tensor_t &bta, const functor_t &fn,
        const permutation<N> &p, const scalar_tr_t &c) :

        m_bta(bta), m_fn(fn), m_tr(p, c), m_bis(mk_bis(m_bta.get_bis(), p)),
        m_bidims(m_bis.get_block_index_dims()), m_sym(m_bis), m_sch(m_bidims) {

    //! Type of block tensor control object
    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    so_apply<N, element_t>(ctrla.req_const_symmetry(), m_tr.get_perm(),
            m_fn.transf(true), m_fn.transf(false),
            m_fn.keep_zero()).perform(m_sym);
    make_schedule();
}


template<size_t N, typename Traits>
void bto_apply<N, Traits>::sync_on() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_on();
}


template<size_t N, typename Traits>
void bto_apply<N, Traits>::sync_off() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    ctrla.req_sync_off();
}


template<size_t N, typename Traits>
void bto_apply<N, Traits>::compute_block(bool zero, block_t &blk,
        const index<N> &ib, const tensor_tr_t &tr, const element_t &c,
        cpu_pool &cpus) {

    static const char *method =
            "compute_block(bool, block_t &, const index<N> &, "
            "const tensor_tr_t &, const scalar_tr_t&, cpu_pool&)";

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template tensor_type<N>::type tensor_t;
    typedef typename Traits::template to_set_type<N>::type to_set_t;
    typedef typename Traits::template to_copy_type<N>::type to_copy_t;
    typedef typename Traits::template to_apply_type<N>::type to_apply_t;

    if(zero) to_set_t().perform(cpus, blk);

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr.get_perm(), true);

    //  Corresponding index in A
    index<N> ia(ib);
    ia.permute(pinv);

    // Find the orbit the index belongs to
    orbit<N, element_t> oa(ctrla.req_const_symmetry(), ia);

    // If the orbit of A is not allowed, we assume it all elements are 0.0
    if (! oa.is_allowed()) {
        if (! m_fn.keep_zero()) {
            tensor_t tblk(blk.get_dims());
            to_set_t(m_fn(Traits::zero()) * c).perform(cpus, tblk);
            to_copy_t(tblk).perform(cpus, false,
                    scalar_tr_t().get_coeff(), blk);
        }
        return;
    }

    //  Find the canonical index in A
    abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);

    //  Transformation for block from canonical A to B
    const tensor_tr_t &tra = oa.get_transf(ia);
    permutation<N> pa(tra.get_perm());
    pa.permute(m_tr.get_perm()).permute(permutation<N>(tr.get_perm(), true));
    scalar_tr_t sa(tra.get_scalar_tr()), sb(c);
    sa.transform(m_tr.get_scalar_tr());
    sb.transform(scalar_tr_t(tr.get_scalar_tr()).invert());

    if(! ctrla.req_is_zero_block(acia.get_index())) {

        block_t &blka = ctrla.req_block(acia.get_index());
        to_apply_t(blka, m_fn, pa, sa.get_coeff()).perform(cpus,
                false, sb.get_coeff(), blk);
        ctrla.ret_block(acia.get_index());
    }
    else {
        if (! m_fn.keep_zero()) {
            tensor_t tblk(blk.get_dims());
            to_set_t(m_fn(Traits::zero()) * c).perform(cpus, tblk);
            to_copy_t(tblk).perform(cpus, false, sb.get_coeff(), blk);
        }
    }
}


template<size_t N, typename Traits>
void bto_apply<N, Traits>::make_schedule() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    block_tensor_ctrl_t ctrla(m_bta);
    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();

    permutation<N> pinv(m_tr.get_perm(), ! m_tr.get_perm().is_identity());

    orbit_list<N, element_t> ol(m_sym);
    for(typename orbit_list<N, element_t>::iterator io = ol.begin();
            io != ol.end(); io++) {

        // If m_fn(0.0) yields 0.0 only non-zero blocks of tensor A need to
        // be considered
        if (m_fn.keep_zero()) {
            index<N> ia(ol.get_index(io)); ia.permute(pinv);

            orbit<N, element_t> oa(ctrla.req_const_symmetry(), ia);
            if (! oa.is_allowed()) continue;

            abs_index<N> acia(oa.get_abs_canonical_index(), bidimsa);
            if (ctrla.req_is_zero_block(acia.get_index())) continue;

            m_sch.insert(ol.get_abs_index(io));
        }
        else {
            m_sch.insert(ol.get_abs_index(io));
        }
    }
}


template<size_t N, typename Traits>
block_index_space<N> bto_apply<N, Traits>::mk_bis(
        const block_index_space<N> &bis, const permutation<N> &perm) {

    block_index_space<N> bis1(bis);
    bis1.permute(perm);
    return bis1;
}



} // namespace libtensor

#endif // LIBTENSOR_BTO_APPLY_IMPL_H
