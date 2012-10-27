#ifndef LIBTENSOR_GEN_BTO_ADD_IMPL_H
#define LIBTENSOR_GEN_BTO_ADD_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirsum.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/symmetry/so_permute.h>
#include <libtensor/core/bad_block_index_space.h>
#include "gen_bto_copy_bis.h"
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_add.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_add<N, Traits, Timed>::k_clazz =
    "gen_bto_add<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
class gen_bto_add_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_add<N, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    index<N> m_idx;
    gen_block_stream_i<N, bti_traits> &m_out;

public:
    gen_bto_add_task(
        gen_bto_add<N, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        const index<N> &idx,
        gen_block_stream_i<N, bti_traits> &out);

    virtual ~gen_bto_add_task() { }
    virtual void perform();

};


template<size_t N, typename Traits, typename Timed>
class gen_bto_add_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;
    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

private:
    gen_bto_add<N, Traits, Timed> &m_bto;
    temp_block_tensor_type &m_btb;
    gen_block_stream_i<N, bti_traits> &m_out;
    const assignment_schedule<N, element_type> &m_sch;
    typename assignment_schedule<N, element_type>::iterator m_i;

public:
    gen_bto_add_task_iterator(
        gen_bto_add<N, Traits, Timed> &bto,
        temp_block_tensor_type &btb,
        gen_block_stream_i<N, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_add_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, typename Traits, typename Timed>
gen_bto_add<N, Traits, Timed>::gen_bto_add(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra) :

    m_bisb(gen_bto_copy_bis<N>(bta.get_bis(), tra.get_perm()).get_bisb()),
    m_symb(m_bisb),
    m_schb(m_bisb.get_block_index_dims()),
    m_valid_sch(false) {

    m_bisb.match_splits();
    add_operand(bta, tra);
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::add_op(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra) {

    static const char *method = "add_op("
        "gen_block_tensor_rd_i<N, bti_traits>&, "
        "const tensor_transf<N, element_type>&)";

    block_index_space<N> bisa(bta.get_bis());
    bisa.permute(tra.get_perm());
    bisa.match_splits();
    if(!m_bisb.equals(bisa)) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bta");
    }

    if(!tra.get_scalar_tr().is_zero()) {
        add_operand(bta, tra);
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::perform(
    gen_block_stream_i<N, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

    gen_bto_add::start_timer();

    try {

        out.open();

        temp_block_tensor_type btb(m_bisb);

        gen_bto_add_task_iterator<N, Traits, Timed> ti(*this, btb, out);
        gen_bto_add_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_add::stop_timer();
        throw;
    }

    gen_bto_add::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::compute_block(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    gen_bto_add::start_timer("compute_block");

    try {

        compute_block_untimed(zero, ib, trb, blkb);

    } catch(...) {
        gen_bto_add::stop_timer("compute_block");
        throw;
    }

    gen_bto_add::stop_timer("compute_block");
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::compute_block_untimed(
    bool zero,
    const index<N> &ib,
    const tensor_transf<N, element_type> &trb,
    wr_block_type &blkb) {

    typedef typename Traits::template to_set_type<N>::type to_set;
    typedef typename Traits::template to_copy_type<N>::type to_copy;

    bool zero1 = zero;

    for(typename std::list<arg>::iterator i = m_args.begin();
        i != m_args.end(); ++i) {

        gen_block_tensor_rd_i<N, bti_traits> &bta = i->bta;
        const tensor_transf<N, element_type> &tra = i->tra;
        tensor_transf<N, element_type> trainv(tra, true);

        gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);

        //  Corresponding index in A
        index<N> ia(ib);
        ia.permute(trainv.get_perm());

        //  Canonical index in A
        orbit<N, element_type> oa(ca.req_const_symmetry(), ia);
        const index<N> &cia = oa.get_cindex();

        if(!oa.is_allowed()) continue;

        //  Transformation for block from canonical A to B
        //  B = c Tr(Bc->B) Tr(A->Bc) Tr(Ac->A) Ac
        tensor_transf<N, element_type> tra1(oa.get_transf(ia));
        tra1.transform(tra).transform(trb);

        //  Compute block in B
        if(!ca.req_is_zero_block(cia)) {
            rd_block_type &blka = ca.req_const_block(cia);
            to_copy(blka, tra1).perform(zero1, blkb);
            ca.ret_const_block(cia);
            zero1 = false;
        }
    }

    if(zero1) {
        to_set().perform(blkb);
    }
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::add_operand(
    gen_block_tensor_rd_i<N, bti_traits> &bta,
    const tensor_transf<N, element_type> &tra) {

    bool first = m_args.empty();
    m_args.push_back(arg(bta, tra));

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);

    if(first) {

        so_permute<N, element_type>(ca.req_const_symmetry(), tra.get_perm()).
            perform(m_symb);

    } else {

        sequence<N, size_t> seq2a;
        sequence<N + N, size_t> seq1b, seq2b;
        for (size_t i = 0; i < N; i++) {
            seq2a[i] = i + N;
            seq1b[i] = seq2b[i] = i;
        }
        tra.get_perm().apply(seq2a);
        for (size_t i = N; i < N + N; i++) {
            seq1b[i] = i; seq2b[i] = seq2a[i - N];
        }
        permutation_builder<N + N> pb(seq2b, seq1b);

        block_index_space_product_builder<N, N> bbx(m_bisb, m_bisb,
            pb.get_perm());

        symmetry<N + N, element_type> symx(bbx.get_bis());
        so_dirsum<N, N, element_type>(m_symb, ca.req_const_symmetry(),
            pb.get_perm()).perform(symx);
        mask<N + N> msk;
        sequence<N + N, size_t> seq;
        for(size_t i = 0; i < N; i++) {
            msk[i] = msk[seq2a[i]] = true;
            seq[i] = seq[seq2a[i]] = i;
        }
        so_merge<N + N, N, element_type>(symx, msk, seq).perform(m_symb);

    }

    m_valid_sch = false;
    //make_schedule();
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add<N, Traits, Timed>::make_schedule() const {

    gen_bto_add::start_timer("make_schedule");

    try {

        orbit_list<N, element_type> olb(m_symb);

        for(typename orbit_list<N, element_type>::iterator iob = olb.begin();
            iob != olb.end(); ++iob) {

            if(m_schb.contains(olb.get_abs_index(iob))) continue;

            for(typename std::list<arg>::const_iterator i = m_args.begin();
                i != m_args.end(); ++i) {

                gen_block_tensor_rd_i<N, bti_traits> &bta = i->bta;
                const tensor_transf<N, element_type> &tra = i->tra;
                tensor_transf<N, element_type> trainv(tra, true);

                gen_block_tensor_rd_ctrl<N, bti_traits> ca(bta);

                index<N> ia(olb.get_index(iob));
                ia.permute(trainv.get_perm());
                orbit<N, element_type> oa(ca.req_const_symmetry(), ia);
                if(!oa.is_allowed()) continue;

                if(!ca.req_is_zero_block(oa.get_cindex())) {
                    m_schb.insert(olb.get_abs_index(iob));
                    break;
                }
            }
        }

    } catch(...) {
        gen_bto_add::stop_timer("make_schedule");
        throw;
    }

    gen_bto_add::stop_timer("make_schedule");
    
    m_valid_sch = true;
}


template<size_t N, typename Traits, typename Timed>
gen_bto_add_task<N, Traits, Timed>::gen_bto_add_task(
    gen_bto_add<N, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    const index<N> &idx,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_idx(idx), m_out(out) {

}


template<size_t N, typename Traits, typename Timed>
void gen_bto_add_task<N, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;
    typedef tensor_transf<N, element_type> tensor_transf_type;

    tensor_transf<N, element_type> tr0;
    gen_block_tensor_ctrl<N, bti_traits> cb(m_btb);

    {
        wr_block_type &blkb = cb.req_block(m_idx);
        m_bto.compute_block_untimed(true, m_idx, tr0, blkb);
        cb.ret_block(m_idx);
    }

    {
        rd_block_type &blkb = cb.req_const_block(m_idx);
        m_out.put(m_idx, blkb, tr0);
        cb.ret_const_block(m_idx);
    }

    cb.req_zero_block(m_idx);
}


template<size_t N, typename Traits, typename Timed>
gen_bto_add_task_iterator<N, Traits, Timed>::gen_bto_add_task_iterator(
    gen_bto_add<N, Traits, Timed> &bto,
    temp_block_tensor_type &btb,
    gen_block_stream_i<N, bti_traits> &out) :

    m_bto(bto), m_btb(btb), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_add_task_iterator<N, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_add_task_iterator<N, Traits, Timed>::get_next() {

    dimensions<N> bidimsb = m_btb.get_bis().get_block_index_dims();
    index<N> idx;
    abs_index<N>::get_index(m_sch.get_abs_index(m_i), bidimsb, idx);
    gen_bto_add_task<N, Traits, Timed> *t =
        new gen_bto_add_task<N, Traits, Timed>(m_bto, m_btb, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void gen_bto_add_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_ADD_IMPL_H
