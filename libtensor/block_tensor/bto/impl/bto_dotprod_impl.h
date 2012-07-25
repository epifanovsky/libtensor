#ifndef LIBTENSOR_BTOD_DOTPROD_IMPL_H
#define LIBTENSOR_BTOD_DOTPROD_IMPL_H

#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>

namespace libtensor {


template<size_t N, typename Traits>
const char *bto_dotprod<N, Traits>::k_clazz = "bto_dotprod<N, Traits>";


template<size_t N, typename Traits>
bto_dotprod<N, Traits>::bto_dotprod(block_tensor_t &bt1,
        block_tensor_t &bt2) : m_bis(bt1.get_bis()) {

    m_bis.match_splits();
    add_arg(bt1, bt2);
}


template<size_t N, typename Traits>
bto_dotprod<N, Traits>::bto_dotprod(block_tensor_t &bt1,
        const permutation<N> &perm1, block_tensor_t &bt2,
        const permutation<N> &perm2) : m_bis(bt1.get_bis()) {

    m_bis.match_splits();
    m_bis.permute(perm1);
    add_arg(bt1, perm1, bt2, perm2);
}


template<size_t N, typename Traits>
void bto_dotprod<N, Traits>::add_arg(block_tensor_t &bt1,
        block_tensor_t &bt2) {

    static const char *method = "add_arg(block_tensor_t&, "
            "block_tensor_t&)";

    block_index_space<N> bis1(bt1.get_bis()), bis2(bt2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    if(!m_bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt1");
    }
    if(!m_bis.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt2");
    }

    m_args.push_back(arg(bt1, bt2));
}


template<size_t N, typename Traits>
void bto_dotprod<N, Traits>::add_arg(block_tensor_t &bt1,
        const permutation<N> &perm1, block_tensor_t &bt2,
        const permutation<N> &perm2) {

    static const char *method = "add_arg(block_tensor_t&, "
            "const permutation<N>&, block_tensor_t&, "
            "const permutation<N>&)";

    block_index_space<N> bis1(bt1.get_bis()), bis2(bt2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    bis1.permute(perm1);
    bis2.permute(perm2);
    if(!m_bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt1");
    }
    if(!m_bis.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt2");
    }

    m_args.push_back(arg(bt1, perm1, bt2, perm2));
}


template<size_t N, typename Traits>
typename Traits::element_type bto_dotprod<N, Traits>::calculate() {

    std::vector<element_t> v(1);
    calculate(v);
    return v[0];
}


template<size_t N, typename Traits>
void bto_dotprod<N, Traits>::calculate(std::vector<element_t> &v) {

    static const char *method = "calculate(std::vector<element_t>&)";

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template to_dotprod_type<N>::type
        to_dotprod_t;

    size_t narg = m_args.size(), i;

    if(v.size() != narg) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "v");
    }

    bto_dotprod<N, Traits>::start_timer();

    try {

        dimensions<N> bidims(m_bis.get_block_index_dims());

        std::vector< block_tensor_ctrl_t* > ctrl1(narg), ctrl2(narg);
        std::vector< symmetry<N, element_t>* > sym(narg);
        std::vector< to_dotprod_t* > tod(narg, (to_dotprod_t*)0);

        typename std::list<arg>::const_iterator j;

        for(i = 0, j = m_args.begin(); i < narg; i++, j++) {
            v[i] = 0.0;
            ctrl1[i] = new block_tensor_ctrl_t(j->bt1);
            ctrl2[i] = new block_tensor_ctrl_t(j->bt2);
            sym[i] = new symmetry<N, element_t>(block_index_space<N>(
                    j->bt1.get_bis()).permute(j->perm1));

            sequence<N, size_t> seq1a, seq2a;
            for (register size_t ii = 0; ii < N; ii++) {
                seq1a[ii] = ii; seq2a[ii] = ii + N;
            }
            j->perm1.apply(seq1a);
            j->perm2.apply(seq2a);
            sequence<N + N, size_t> seq1b, seq2b;
            for (register size_t ii = 0; ii < N; ii++) {
                seq1b[ii] = ii; seq2b[ii] = seq1a[ii];
            }
            for (register size_t ii = N; ii < N + N; ii++) {
                seq1b[ii] = ii; seq2b[ii] = seq2a[ii - N];
            }
            permutation_builder<N + N> pbb(seq2b, seq1b);

            block_index_space_product_builder<N, N> bbx(j->bt1.get_bis(),
                    j->bt2.get_bis(), pbb.get_perm());

            symmetry<N + N, element_t> symx(bbx.get_bis());
            so_dirprod<N, N, element_t>(ctrl1[i]->req_const_symmetry(),
                    ctrl2[i]->req_const_symmetry(),
                    pbb.get_perm()).perform(symx);

            mask<N + N> msk;
            sequence<N + N, size_t> seq;
            for (register size_t ii = 0; ii < N; ii++) {
                msk[ii] = msk[ii + N] = true;
                seq[ii] = seq[ii + N] = ii;
            }
            so_merge<N + N, N, element_t>(symx, msk, seq).perform(*sym[i]);
        }

        for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

            orbit_list<N, element_t> ol1(ctrl1[i]->req_const_symmetry());
            orbit_list<N, element_t> ol2(ctrl2[i]->req_const_symmetry());
            orbit_list<N, element_t> ol(*sym[i]);

            permutation<N> pinv1(j->perm1, true), pinv2(j->perm2, true);

            ctrl1[i]->req_sync_on();
            ctrl2[i]->req_sync_on();

            std::vector<dotprod_in_orbit_task*> tasklist;

            for(typename orbit_list<N, element_t>::iterator io = ol.begin();
                    io != ol.end(); io++) {

                index<N> i1(ol.get_index(io)), i2(ol.get_index(io));
                i1.permute(pinv1);
                i2.permute(pinv2);

                dotprod_in_orbit_task *t = new dotprod_in_orbit_task(
                        j->bt1, ol1, pinv1, j->bt2, ol2, pinv2,
                        *sym[i], bidims, ol.get_index(io));
                tasklist.push_back(t);
            }

            dotprod_task_iterator ti(tasklist);
            dotprod_task_observer to;
            libutil::thread_pool::submit(ti, to);

            for(size_t k = 0; k < tasklist.size(); k++) {
                v[i] += tasklist[k]->get_d();
                delete tasklist[k];
            }

            ctrl1[i]->req_sync_off();
            ctrl2[i]->req_sync_off();
        }

        for(i = 0; i < narg; i++) {
            delete sym[i];
            delete ctrl1[i];
            delete ctrl2[i];
        }

    } catch(...) {
        bto_dotprod<N, Traits>::stop_timer();
        throw;
    }

    bto_dotprod<N, Traits>::stop_timer();
}


template<size_t N, typename Traits>
const char *bto_dotprod<N, Traits>::dotprod_in_orbit_task::k_clazz =
        "bto_dotprod<N, Traits>::dotprod_in_orbit_task";


template<size_t N, typename Traits>
void bto_dotprod<N, Traits>::dotprod_in_orbit_task::perform() {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template block_type<N>::type block_t;
    typedef typename Traits::template to_dotprod_type<N>::type
        to_dotprod_t;

    block_tensor_ctrl_t ctrl1(m_bt1), ctrl2(m_bt2);

    orbit<N, element_t> orb(m_sym, m_idx);
    element_t c = Traits::zero();
    for(typename orbit<N, element_t>::iterator io = orb.begin();
            io != orb.end(); io++)
        c += orb.get_transf(io).get_scalar_tr().get_coeff();

    if(Traits::is_zero(c)) return;

    dimensions<N> bidims1(m_bidims), bidims2(m_bidims);
    bidims1.permute(m_pinv1);
    bidims2.permute(m_pinv2);

    index<N> i1(m_idx), i2(m_idx);
    i1.permute(m_pinv1);
    i2.permute(m_pinv2);

    orbit<N, element_t> orb1(ctrl1.req_const_symmetry(), i1),
            orb2(ctrl2.req_const_symmetry(), i2);

    const tensor_transf<N, element_t> &tr1 = orb1.get_transf(i1);
    const tensor_transf<N, element_t> &tr2 = orb2.get_transf(i2);

    abs_index<N> aci1(orb1.get_abs_canonical_index(), bidims1),
            aci2(orb2.get_abs_canonical_index(), bidims2);
    if(ctrl1.req_is_zero_block(aci1.get_index()) ||
            ctrl2.req_is_zero_block(aci2.get_index())) return;

    block_t &blk1 = ctrl1.req_block(aci1.get_index());
    block_t &blk2 = ctrl2.req_block(aci2.get_index());

    permutation<N> perm1, perm2;
    perm1.permute(tr1.get_perm()).permute(permutation<N>(m_pinv1, true));
    perm2.permute(tr2.get_perm()).permute(permutation<N>(m_pinv2, true));

    element_t d = to_dotprod_t(blk1, perm1, blk2, perm2).calculate() *
            tr1.get_scalar_tr().get_coeff() * tr2.get_scalar_tr().get_coeff();

    ctrl1.ret_block(aci1.get_index());
    ctrl2.ret_block(aci2.get_index());

    m_d = c * d;
}


template<size_t N, typename Traits>
bool bto_dotprod<N, Traits>::dotprod_task_iterator::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, typename Traits>
libutil::task_i *bto_dotprod<N, Traits>::dotprod_task_iterator::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_DOTPROD_IMPL_H
