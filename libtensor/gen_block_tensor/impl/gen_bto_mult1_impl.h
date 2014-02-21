#ifndef LIBTENSOR_GEN_BTO_MULT1_IMPL_H
#define LIBTENSOR_GEN_BTO_MULT1_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include "../addition_schedule.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_aux_add.h"
#include "../gen_bto_mult1.h"
#include "gen_bto_copy_impl.h"

namespace libtensor {


namespace {


template<size_t N, typename Traits, typename Timed>
class gen_bto_mult1_task : public libutil::task_i, public timings<Timed> {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_i<N, bti_traits> &m_bta;
    gen_block_tensor_rd_i<N, bti_traits> &m_btb;
    tensor_transf<N, element_type> m_trb;
    scalar_transf<element_type> m_c;
    index<N> m_idxa, m_idxb;
    bool m_recip, m_zero;

public:
    gen_bto_mult1_task(bool zero,
        gen_block_tensor_i<N, bti_traits> &bta, const index<N> &idxa,
        gen_block_tensor_rd_i<N, bti_traits> &btb, const index<N> &idxb,
        const tensor_transf<N, element_type> &trb, bool recip,
        const scalar_transf<element_type> &c);

    virtual ~gen_bto_mult1_task() { }
    virtual unsigned long get_cost() const { return 0; }
    virtual void perform();

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_mult1_task_iterator : public libutil::task_iterator_i {
public:
    typedef gen_bto_mult1_task<N, Traits, Timed> task_type;

private:
    std::vector<task_type *> m_tl;
    typename std::vector<task_type *>::iterator m_i;

public:
    gen_bto_mult1_task_iterator(std::vector<task_type *> &tl) :
        m_tl(tl), m_i(m_tl.begin()) {

    }

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_mult1_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


} // unnamed namespace


template<size_t N, typename Traits, typename Timed>
const char gen_bto_mult1<N, Traits, Timed>::k_clazz[] =
    "gen_bto_mult1<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_mult1<N, Traits, Timed>::gen_bto_mult1(
    gen_block_tensor_rd_i<N, bti_traits> &btb,
    const tensor_transf_type &trb, bool recip,
    const scalar_transf<element_type> &c) :

    m_btb(btb), m_trb(trb), m_recip(recip), m_c(c) {

    static const char method[] = "gen_bto_mult1()";

    if(m_recip && m_trb.get_scalar_tr().is_zero()) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "trb");
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_mult1<N, Traits, Timed>::perform(
    bool zero,
    gen_block_tensor_i<N, bti_traits> &bta) {

    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_a_type;
    typedef gen_bto_mult1_task<N, Traits, Timed> task_type;

    static const char method[] =
        "perform(bool, gen_block_tensor_i<N, bti_traits>&)";

    if(!bta.get_bis().equals(m_btb.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bta");
    }

    gen_bto_mult1::start_timer();

    try {

        {
            //  Make empty tensor A1 with the new symmetry of A

            temp_block_tensor_a_type bta1(bta.get_bis());

            gen_block_tensor_ctrl<N, bti_traits> ca(bta);
            gen_block_tensor_wr_ctrl<N, bti_traits> ca1(bta1);
            gen_block_tensor_rd_ctrl<N, bti_traits> cb(m_btb);

            sequence<N + N, size_t> seq1b, seq2b;
            for(size_t i = 0; i < N; i++) seq1b[i] = seq2b[i] = i;
            for(size_t i = N, j = 0; i < N + N; i++, j++) {
                seq1b[i] = i; seq2b[i] = m_trb.get_perm()[j] + N;
            }
            permutation_builder<N + N> pbb(seq2b, seq1b);

            block_index_space_product_builder<N, N> bbx(bta.get_bis(),
                bta.get_bis(), permutation<N + N>());

            symmetry<N + N, element_type> symx(bbx.get_bis());
            so_dirprod<N, N, element_type>(ca.req_const_symmetry(),
                cb.req_const_symmetry(), pbb.get_perm()).perform(symx);
            mask<N + N> msk;
            sequence<N + N, size_t> seq;
            for(register size_t i = 0; i < N; i++) {
                msk[i] = msk[i + N] = true;
                seq[i] = seq[i + N] = i;
            }
            so_merge<N + N, N, element_type>(symx, msk, seq).
                perform(ca1.req_symmetry());

            //  Install new symmetry into A

            gen_bto_copy<N, Traits, Timed> copy(bta1,
                tensor_transf<N, element_type>());

            std::vector<size_t> nzblka;
            ca.req_nonzero_blocks(nzblka);
            addition_schedule<N, Traits> asch(copy.get_symmetry(),
                ca.req_const_symmetry());
            asch.build(copy.get_schedule(), nzblka);

            gen_bto_aux_add<N, Traits> out(copy.get_symmetry(), asch, bta,
                scalar_transf<element_type>());
            out.open();
            copy.perform(out);
            out.close();
        }

        std::vector<task_type*> tasklist;

        {
            gen_block_tensor_ctrl<N, bti_traits> ca(bta);
            gen_block_tensor_rd_ctrl<N, bti_traits> cb(m_btb);

            dimensions<N> bidimsa(bta.get_bis().get_block_index_dims());
            dimensions<N> bidimsb(m_btb.get_bis().get_block_index_dims());
            permutation<N> pinvb(m_trb.get_perm(), true);

            const symmetry<N, element_type> &symb = cb.req_const_symmetry();

            std::vector<size_t> nzblka;
            ca.req_nonzero_blocks(nzblka);

            for(size_t iblka = 0; iblka < nzblka.size(); iblka++) {

                index<N> idxa;
                abs_index<N>::get_index(nzblka[iblka], bidimsa, idxa);

                index<N> idxb(idxa);
                idxb.permute(pinvb);

                orbit<N, element_type> ob(symb, idxb);

                bool zerob = cb.req_is_zero_block(ob.get_cindex());
                if(zerob) {
                    if(m_recip) {
                        throw bad_parameter(g_ns, k_clazz, method,
                            __FILE__, __LINE__, "zero in btb");
                    }
                    if(zero) ca.req_zero_block(idxa);
                    continue;
                }

                tensor_transf_type trb(ob.get_transf(idxb));
                trb.transform(m_trb);

                task_type *task = new task_type(zero, bta, idxa,
                    m_btb, ob.get_cindex(), trb, m_recip, m_c);
                tasklist.push_back(task);
        }
        }

        gen_bto_mult1_task_iterator<N, Traits, Timed> ti(tasklist);
        gen_bto_mult1_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch (...) {
        gen_bto_mult1::stop_timer();
        throw;
    }

    gen_bto_mult1::stop_timer();
}


namespace {


template<size_t N, typename Traits, typename Timed>
gen_bto_mult1_task<N, Traits, Timed>::gen_bto_mult1_task(bool zero,
    gen_block_tensor_i<N, bti_traits> &bta, const index<N> &idxa,
    gen_block_tensor_rd_i<N, bti_traits> &btb, const index<N> &idxb,
    const tensor_transf<N, element_type> &trb, bool recip,
    const scalar_transf<element_type> &c) :

    m_bta(bta), m_btb(btb), m_trb(trb), m_c(c), m_idxa(idxa), m_idxb(idxb),
    m_recip(recip), m_zero(zero)  {

}

template<size_t N, typename Traits, typename Timed>
void gen_bto_mult1_task<N, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type
            rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type
            wr_block_type;
    typedef typename Traits::template to_mult1_type<N>::type to_mult1;

    gen_bto_mult1_task::start_timer();

    gen_block_tensor_ctrl<N, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<N, bti_traits> cb(m_btb);

    wr_block_type &blka = ca.req_block(m_idxa);
    rd_block_type &blkb = cb.req_const_block(m_idxb);

    to_mult1(blkb, m_trb, m_recip, m_c).perform(m_zero, blka);

    cb.ret_const_block(m_idxb);
    ca.ret_block(m_idxa);

    gen_bto_mult1_task::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_mult1_task_iterator<N, Traits, Timed>::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_mult1_task_iterator<N, Traits, Timed>::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void gen_bto_mult1_task_observer<N, Traits>::notify_finish_task(
        libutil::task_i *t) {

    delete t;
}


} // unnamed namespace


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_MULT1_IMPL_H
