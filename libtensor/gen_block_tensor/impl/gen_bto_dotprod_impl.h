#ifndef LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H
#define LIBTENSOR_GEN_BTO_DOTPROD_IMPL_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/core/subgroup_orbits.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include "../gen_bto_dotprod.h"

namespace libtensor {


namespace {


template<size_t N, typename Traits, typename Timed>
class gen_bto_dotprod_task : public libutil::task_i, public timings<Timed> {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bta;
    tensor_transf<N, element_type> m_tra;
    gen_block_tensor_rd_i<N, bti_traits> &m_btb;
    tensor_transf<N, element_type> m_trb;
    const symmetry<N, element_type> &m_symat;
    const symmetry<N, element_type> &m_symc;
    dimensions<N> m_bidimsc;
    size_t m_aidxa;
    element_type m_d;

public:
    gen_bto_dotprod_task(
        gen_block_tensor_rd_i<N, bti_traits> &bta,
        const tensor_transf<N, element_type> &tra,
        gen_block_tensor_rd_i<N, bti_traits> &btb,
        const tensor_transf<N, element_type> &trb,
        const symmetry<N, element_type> &symat,
        const symmetry<N, element_type> &symc,
        const dimensions<N> &bidimsc,
        size_t aidxa) :

        m_bta(bta), m_tra(tra), m_btb(btb), m_trb(trb), m_symat(symat),
        m_symc(symc), m_bidimsc(bidimsc), m_aidxa(aidxa), m_d(Traits::zero())
    { }

    virtual ~gen_bto_dotprod_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

    element_type get_d() const { return m_d; }

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_dotprod_task_iterator : public libutil::task_iterator_i {
public:
    typedef gen_bto_dotprod_task<N, Traits, Timed> task_type;

private:
    std::vector<task_type*> &m_tl;
    typename std::vector<task_type*>::iterator m_i;

public:
    gen_bto_dotprod_task_iterator(std::vector<task_type*> &tl) :
        m_tl(tl), m_i(m_tl.begin())
    { }

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


} // unnamed namespace


template<size_t N, typename Traits, typename Timed>
const char gen_bto_dotprod<N, Traits, Timed>::k_clazz[] =
    "gen_bto_dotprod<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_dotprod<N, Traits, Timed>::gen_bto_dotprod(
    gen_block_tensor_rd_i<N, bti_traits> &bt1,
    const tensor_transf_type &tr1,
    gen_block_tensor_rd_i<N, bti_traits> &bt2,
    const tensor_transf_type &tr2) :

    m_bis(bt1.get_bis()) {

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

    static const char method[] = "add_arg()";

    block_index_space<N> bis1(bt1.get_bis()), bis2(bt2.get_bis());
    bis1.match_splits();
    bis2.match_splits();
    bis1.permute(tr1.get_perm());
    bis2.permute(tr2.get_perm());
    if(!m_bis.equals(bis1)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt1");
    }
    if(!m_bis.equals(bis2)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bt2");
    }

    m_args.push_back(arg(bt1, tr1, bt2, tr2));
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod<N, Traits, Timed>::calculate(
    std::vector<element_type> &v) {

    static const char method[] = "calculate(std::vector<element_type>&)";

    typedef gen_bto_dotprod_task<N, Traits, Timed> task_type;

    if(v.size() != m_args.size()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "v");
    }

    gen_bto_dotprod::start_timer();

    try {

        dimensions<N> bidims(m_bis.get_block_index_dims());

        typename std::list<arg>::const_iterator iarg = m_args.begin();
        for(size_t i = 0; iarg != m_args.end(); i++, ++iarg) {

            v[i] = Traits::zero();

            gen_block_tensor_rd_i<N, bti_traits> &bta = iarg->bt1;
            gen_block_tensor_rd_i<N, bti_traits> &btb = iarg->bt2;
            const tensor_transf<N, element_type> &tra = iarg->tr1;
            const tensor_transf<N, element_type> &trb = iarg->tr2;
            dimensions<N> bidimsa = bta.get_bis().get_block_index_dims();

            gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);
            gen_block_tensor_rd_ctrl<N, bti_traits> cb(btb);

            const symmetry<N, element_type> &syma = ca.req_const_symmetry();
            const symmetry<N, element_type> &symb = cb.req_const_symmetry();

            symmetry<N, element_type> symat(m_bis), symc(m_bis);

            sequence<N, size_t> seq1a, seq2a;
            for(size_t ii = 0; ii < N; ii++) {
                seq1a[ii] = ii;
                seq2a[ii] = N + ii;
            }
            tra.get_perm().apply(seq1a);
            trb.get_perm().apply(seq2a);
            sequence<N + N, size_t> seq1b, seq2b;
            for(size_t ii = 0; ii < N; ii++) {
                seq1b[ii] = ii;
                seq1b[N + ii] = N + ii;
                seq2b[ii] = seq1a[ii];
                seq2b[N + ii] = seq2a[ii];
            }
            permutation_builder<N + N> pbb(seq2b, seq1b);

            block_index_space_product_builder<N, N> bbx(bta.get_bis(),
                btb.get_bis(), pbb.get_perm());

            symmetry<N + N, element_type> symx(bbx.get_bis());
            so_dirprod<N, N, element_type>(syma, symb, pbb.get_perm()).
                perform(symx);

            mask<N + N> msk;
            sequence<N + N, size_t> seq;
            for(size_t ii = 0; ii < N; ii++) {
                msk[ii] = msk[ii + N] = true;
                seq[ii] = seq[ii + N] = ii;
            }
            so_merge<N + N, N, element_type>(symx, msk, seq).perform(symc);

            so_permute<N, element_type>(syma, tra.get_perm()).perform(symat);

            //  Warning! Cannot move tasklist out of this scope because
            //  it uses symat, symc
            std::vector<task_type*> tasklist;

            std::vector<size_t> nzblka;
            ca.req_nonzero_blocks(nzblka);

            for(size_t ia = 0; ia < nzblka.size(); ia++) {
            
                task_type *t = new task_type(bta, tra, btb, trb,
                    symat, symc, bidims, nzblka[ia]);
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

    } catch(...) {
        gen_bto_dotprod::stop_timer();
        throw;
    }

    gen_bto_dotprod::stop_timer();
}


namespace {


template<size_t N, typename Traits, typename Timed>
void gen_bto_dotprod_task<N, Traits, Timed>::perform() {

    typedef typename Traits::template to_dotprod_type<N>::type to_dotprod_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

    m_d = Traits::zero();

    dimensions<N> bidimsa = m_bta.get_bis().get_block_index_dims();
    dimensions<N> bidimsb = m_btb.get_bis().get_block_index_dims();

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta), cb(m_btb);

    const symmetry<N, element_type> &syma = ca.req_const_symmetry();
    const symmetry<N, element_type> &symb = cb.req_const_symmetry();

    permutation<N> pinva(m_tra.get_perm(), true), pinvb(m_trb.get_perm(), true);

    orbit<N, element_type> oa(syma, m_aidxa, true);
    index<N> idxat;
    abs_index<N>::get_index(m_aidxa, bidimsa, idxat);
    idxat.permute(m_tra.get_perm());
    size_t aidxat = abs_index<N>::get_abs_index(idxat, m_bidimsc);

    rd_block_type &blka = ca.req_const_block(oa.get_cindex());

    subgroup_orbits<N, element_type> sgo(m_symat, m_symc, aidxat);
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

        index<N> idxa(idxc), idxb(idxc);
        idxa.permute(pinva);
        idxb.permute(pinvb);

        orbit<N, element_type> ob(symb, idxb, true);
        if(!ob.is_allowed() || cb.req_is_zero_block(ob.get_cindex())) continue;

        tensor_transf<N, element_type> tra(oa.get_transf(idxa)),
            trb(ob.get_transf(idxb));
        tra.transform(m_tra);
        trb.transform(m_trb);

        rd_block_type &blkb = cb.req_const_block(ob.get_cindex());

        element_type d = to_dotprod_type(blka, tra, blkb, trb).calculate();

        cb.ret_const_block(ob.get_cindex());

        sum.apply(d);
        m_d += d;
    }

    ca.ret_const_block(oa.get_cindex());
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


} // unnamed namespace


} // namespace libtensor

#endif // LIBTENSOR_BTO_DOTPROD_IMPL_H
