#ifndef LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H
#define LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/btod/bad_block_index_space.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include "../gen_bto_dotprod.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
class gen_bto_dotprod_in_orbit_task :
public libutil::task_i, public timings<Timed> {

    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

public:
    static const char *k_clazz;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    const orbit_list<N, element_type> &m_ola;
    tensor_transf<N, element_type> m_tra;
    gen_block_tensor_rd_i<N, bti_traits> &m_btb;
    const orbit_list<N, element_type> &m_olb;
    tensor_transf<N, element_type> m_trb;
    const symmetry<N, element_type> &m_symc;
    dimensions<N> m_bidimsc;
    index<N> m_idxc;
    element_type m_d;

public:
    gen_bto_dotprod_in_orbit_task(
            gen_block_tensor_rd_i<N, bti_traits> &bta,
            const orbit_list<N, element_type> &ola,
            const tensor_transf<N, element_type> &tra,
            gen_block_tensor_rd_i<N, bti_traits> &btb,
            const orbit_list<N, element_type> &olb,
            const tensor_transf<N, element_type> &trb,
            const symmetry<N, element_type> &symc,
            const dimensions<N> &bidimsc, const index<N> &idxc) :
        m_bta(bta), m_ola(ola), m_tra(tra),
        m_btb(btb), m_olb(olb), m_trb(trb),
        m_symc(symc), m_bidimsc(bidimsc), m_idxc(idxc), m_d(Traits::zero()) { }

    virtual ~gen_bto_dotprod_in_orbit_task() { }
    virtual void perform();

    element_type get_d() const { return m_d; }
};


template<size_t N, typename Traits, typename Timed>
class gen_bto_dotprod_task_iterator : public libutil::task_iterator_i {
public:
    typedef gen_bto_dotprod_in_orbit_task<N, Traits, Timed> task_type;

private:
    std::vector<task_type *> &m_tl;
    typename std::vector<task_type *>::iterator m_i;

public:
    gen_bto_dotprod_task_iterator(std::vector<task_type *> &tl) :
        m_tl(tl), m_i(m_tl.begin()) {

    }

    virtual ~gen_bto_dotprod_task_iterator() { }

    virtual bool has_more() const;

    virtual libutil::task_i *get_next();
};


template<size_t N, typename Traits>
class gen_bto_dotprod_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) { }
};


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_dotprod<N, Traits, Timed>::k_clazz =
        "gen_bto_dotprod<N, Traits, Timed>";

template<size_t N, typename Traits, typename Timed>
gen_bto_dotprod<N, Traits, Timed>::gen_bto_dotprod(
        gen_block_tensor_rd_i<N, bti_traits> &bt1,
        const tensor_transf_type &tr1,
        gen_block_tensor_rd_i<N, bti_traits> &bt2,
        const tensor_transf_type &tr2) : m_bis(bt1.get_bis()) {

    m_bis.match_splits();
    m_bis.permute(tr1.get_perm());
    add_arg(bt1, tr1, bt2, tr2);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod<N, Traits, Timed>::add_arg(
        gen_block_tensor_rd_i<N, bti_traits> &bt1,
        const tensor_transf_type &tr1,
        gen_block_tensor_rd_i<N, bti_traits> &bt2,
        const tensor_transf_type &tr2) {

    static const char *method = "add_arg("
            "gen_block_tensor_rd_i<N, bti_traits>&, "
            "const tensor_transf_type&, "
            "gen_block_tensor_rd_i<N, bti_traits>&, "
            "const tensor_transf_type&)";

    block_index_space<N> bis1(bt1.get_bis()), bis2(bt2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    bis1.permute(tr1.get_perm());
    bis2.permute(tr2.get_perm());
    if(! m_bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt1");
    }
    if(! m_bis.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bt2");
    }

    m_args.push_back(arg(bt1, tr1, bt2, tr2));
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod<N, Traits, Timed>::calculate(
        std::vector<element_type> &v) {

    static const char *method = "calculate(std::vector<element_type>&)";

    typedef gen_bto_dotprod_in_orbit_task<N, Traits, Timed> task_type;

    typedef typename Traits::template to_dotprod_type<N>::type to_dotprod_type;

    size_t narg = m_args.size(), i;

    if(v.size() != narg) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "v");
    }

    gen_bto_dotprod::start_timer();

    // TODO: Re-iterate over code to construct tasks
    try {

        dimensions<N> bidims(m_bis.get_block_index_dims());

        std::vector<gen_block_tensor_rd_ctrl<N, bti_traits> *> ca(narg),
                cb(narg);
        std::vector<symmetry<N, element_type> *> sym(narg);
        std::vector<to_dotprod_type *> tod(narg, (to_dotprod_type *) 0);

        typename std::list<arg>::const_iterator j;
        for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

            v[i] = Traits::zero();
            ca[i] = new gen_block_tensor_rd_ctrl<N, bti_traits>(j->bt1);
            cb[i] = new gen_block_tensor_rd_ctrl<N, bti_traits>(j->bt2);
            sym[i] = new symmetry<N, element_type>(block_index_space<N>(
                    j->bt1.get_bis()).permute(j->tr1.get_perm()));

            sequence<N, size_t> seq1a, seq2a;
            for (register size_t ii = 0; ii < N; ii++) {
                seq1a[ii] = ii; seq2a[ii] = ii + N;
            }
            j->tr1.get_perm().apply(seq1a);
            j->tr2.get_perm().apply(seq2a);
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

            symmetry<N + N, element_type> symx(bbx.get_bis());
            so_dirprod<N, N, element_type>(ca[i]->req_const_symmetry(),
                    cb[i]->req_const_symmetry(),
                    pbb.get_perm()).perform(symx);

            mask<N + N> msk;
            sequence<N + N, size_t> seq;
            for (register size_t ii = 0; ii < N; ii++) {
                msk[ii] = msk[ii + N] = true;
                seq[ii] = seq[ii + N] = ii;
            }
            so_merge<N + N, N, element_type>(symx, msk, seq).perform(*sym[i]);
        }

        for(i = 0, j = m_args.begin(); i < narg; i++, j++) {

            orbit_list<N, element_type> ol1(ca[i]->req_const_symmetry());
            orbit_list<N, element_type> ol2(cb[i]->req_const_symmetry());
            orbit_list<N, element_type> ol(*sym[i]);

            std::vector<task_type *> tasklist;
            for(typename orbit_list<N, element_type>::iterator io = ol.begin();
                    io != ol.end(); io++) {

                task_type *t = new task_type(j->bt1, ol1, j->tr1,
                        j->bt2, ol2, j->tr2, *sym[i], bidims, ol.get_index(io));
                tasklist.push_back(t);
            }

            gen_bto_dotprod_task_iterator<N, Traits, Timed> ti(tasklist);
            gen_bto_dotprod_task_observer<N, Traits> to;
            libutil::thread_pool::submit(ti, to);

            for(size_t k = 0; k < tasklist.size(); k++) {
                v[i] += tasklist[k]->get_d();
                delete tasklist[k];
            }
        }

        for(i = 0; i < narg; i++) {
            delete sym[i];
            delete ca[i];
            delete cb[i];
        }

    } catch(...) {
        gen_bto_dotprod::stop_timer();
        throw;
    }

    gen_bto_dotprod::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod_in_orbit_task<N, Traits, Timed>::perform() {

    typedef typename Traits::template to_dotprod_type<N>::type to_dotprod_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    gen_bto_dotprod_in_orbit_task::start_timer();

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta), cb(m_btb);

    orbit<N, element_type> orb(m_symc, m_idxc);
    scalar_transf_sum<element_type> sum;
    for(typename orbit<N, element_type>::iterator io = orb.begin();
            io != orb.end(); io++) {
        sum.add(orb.get_transf(io).get_scalar_tr());
    }
    if(sum.is_zero()) return;

    permutation<N> pinva(m_tra.get_perm(), true),
            pinvb(m_trb.get_perm(), true);

    dimensions<N> bidimsa(m_bidimsc), bidimsb(m_bidimsc);
    bidimsa.permute(pinva);
    bidimsb.permute(pinvb);

    index<N> idxa(m_idxc), idxb(m_idxc);
    idxa.permute(pinva);
    idxb.permute(pinvb);

    orbit<N, element_type> orba(ca.req_const_symmetry(), idxa),
            orbb(cb.req_const_symmetry(), idxb);
    abs_index<N> acia(orba.get_abs_canonical_index(), bidimsa),
            acib(orbb.get_abs_canonical_index(), bidimsb);

    if(ca.req_is_zero_block(acia.get_index()) ||
            cb.req_is_zero_block(acib.get_index())) return;

    tensor_transf<N, element_type> tra(orba.get_transf(idxa)),
            trb(orbb.get_transf(idxb));
    tra.transform(m_tra);
    trb.transform(m_trb);

    rd_block_type &blka = ca.req_const_block(acia.get_index());
    rd_block_type &blkb = cb.req_const_block(acib.get_index());

    m_d = to_dotprod_type(blka, tra, blkb, trb).calculate();

    ca.ret_const_block(acia.get_index());
    cb.ret_const_block(acib.get_index());

    sum.apply(m_d);

    gen_bto_dotprod_in_orbit_task::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_dotprod_task_iterator<N, Traits, Timed>::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_dotprod_task_iterator<N, Traits, Timed>::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_DOTPROD_IMPL_H
