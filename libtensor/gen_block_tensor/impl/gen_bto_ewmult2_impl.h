#ifndef LIBTENSOR_GEN_BTO_EWMULT2_IMPL_H
#define LIBTENSOR_GEN_BTO_EWMULT2_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/block_index_space_product_builder.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/symmetry/so_dirprod.h>
#include <libtensor/symmetry/so_merge.h>
#include <libtensor/btod/bad_block_index_space.h>
#include "../gen_block_tensor_ctrl.h"
#include "../gen_bto_ewmult2.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
const char *gen_bto_ewmult2<N, M, K, Traits, Timed>::k_clazz =
        "gen_bto_ewmult2<N, M, K, Traits, Timed>";


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_ewmult2_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_ewmult2<N, M, K, Traits, Timed> &m_gbto;
    gen_block_tensor_i<N + M + K, bti_traits> &m_btc;
    index<N + M + K> m_idx;
    gen_block_stream_i<N + M + K, bti_traits> &m_out;

public:
    gen_bto_ewmult2_task(
        gen_bto_ewmult2<N, M, K, Traits, Timed> &bto,
        gen_block_tensor_i<N + M + K, bti_traits> &btc,
        const index<N + M + K> &idx,
        gen_block_stream_i<N + M + K, bti_traits> &out);

    virtual ~gen_bto_ewmult2_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
class gen_bto_ewmult2_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_ewmult2<N, M, K, Traits, Timed> &m_bto;
    gen_block_tensor_i<N + M + K, bti_traits> &m_btc;
    gen_block_stream_i<N + M + K, bti_traits> &m_out;
    const assignment_schedule<N + M + K, element_type> &m_sch;
    typename assignment_schedule<N + M + K, element_type>::iterator m_i;

public:
    gen_bto_ewmult2_task_iterator(
        gen_bto_ewmult2<N, M, K, Traits, Timed> &bto,
        gen_block_tensor_i<N + M + K, bti_traits> &btc,
        gen_block_stream_i<N + M + K, bti_traits> &out);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K>
class gen_bto_ewmult2_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_ewmult2<N, M, K, Traits, Timed>::gen_bto_ewmult2(
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const tensor_transf<NA, element_type> &tra,
        gen_block_tensor_rd_i<NB, bti_traits> &btb,
        const tensor_transf<NB, element_type> &trb,
        const tensor_transf_type &trc) :

    m_bta(bta), m_tra(tra), m_btb(btb), m_trb(trb), m_trc(trc),
    m_bisc(make_bisc(bta.get_bis(), tra.get_perm(),
            btb.get_bis(), trb.get_perm(), trc.get_perm())),
    m_symc(m_bisc), m_sch(m_bisc.get_block_index_dims()) {

    make_symc();
    make_schedule();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2<N, M, K, Traits, Timed>::perform(
    gen_block_stream_i<NC, bti_traits> &out) {

    typedef typename Traits::template temp_block_tensor_type<NC>::type
        temp_block_tensor_type;

    gen_bto_ewmult2::start_timer();

    try {

        out.open();

        temp_block_tensor_type btc(m_bisc);

        gen_bto_ewmult2_task_iterator<N, M, K, Traits, Timed> ti(*this,
                btc, out);
        gen_bto_ewmult2_task_observer<N, M, K> to;
        libutil::thread_pool::submit(ti, to);

        out.close();

    } catch(...) {
        gen_bto_ewmult2::stop_timer();
        throw;
    }

    gen_bto_ewmult2::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2<N, M, K, Traits, Timed>::compute_block(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf_type &trc,
        wr_block_type &blkc) {

    gen_bto_ewmult2::start_timer();

    try {

        compute_block_untimed(zero, idxc, trc, blkc);

    } catch(...) {
        gen_bto_ewmult2::stop_timer();
        throw;
    }

    gen_bto_ewmult2::stop_timer();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2<N, M, K, Traits, Timed>::compute_block_untimed(
        bool zero,
        const index<NC> &idxc,
        const tensor_transf_type &trc,
        wr_block_type &blkc) {

    typedef typename Traits::template to_set_type<NC>::type to_set;
    typedef typename Traits::template to_ewmult2_type<N, M, K>::type to_ewmult2;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrla(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> ctrlb(m_btb);

    index<NA> idxa;
    index<NB> idxb;
    index<NC> idxstd(idxc);
    idxstd.permute(permutation<NC>(m_trc.get_perm(), true));
    for(size_t i = 0; i < N; i++) idxa[i] = idxstd[i];
    for(size_t i = 0; i < M; i++) idxb[i] = idxstd[N + i];
    for(size_t i = 0; i < K; i++) {
        idxa[N + i] = idxb[M + i] = idxstd[N + M +i];
    }
    idxa.permute(permutation<NA>(m_tra.get_perm(), true));
    idxb.permute(permutation<NB>(m_trb.get_perm(), true));

    orbit<NA, element_type> oa(ctrla.req_const_symmetry(), idxa);
    orbit<NB, element_type> ob(ctrlb.req_const_symmetry(), idxb);

    index<NA> idxa0;
    abs_index<NA>::get_index(oa.get_abs_canonical_index(),
        m_bta.get_bis().get_block_index_dims(), idxa0);
    tensor_transf<NA, element_type> tra(oa.get_transf(idxa));
    tra.transform(m_tra);

    index<NB> idxb0;
    abs_index<NB>::get_index(ob.get_abs_canonical_index(),
        m_btb.get_bis().get_block_index_dims(), idxb0);
    tensor_transf<NB, element_type> trb(ob.get_transf(idxb));
    trb.transform(m_trb);

    bool zeroa = ctrla.req_is_zero_block(idxa0);
    bool zerob = ctrlb.req_is_zero_block(idxb0);

    if(zeroa || zerob) {
        if(zero) to_set().perform(blkc);
        return;
    }

    rd_block_a_type &blka = ctrla.req_const_block(idxa0);
    rd_block_b_type &blkb = ctrlb.req_const_block(idxb0);

    tensor_transf_type tr(m_trc);
    tr.transform(trc);
    to_ewmult2(blka, tra, blkb, trb, tr).perform(zero, blkc);

    ctrla.ret_const_block(idxa0);
    ctrlb.ret_const_block(idxb0);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
block_index_space<N + M + K>
gen_bto_ewmult2<N, M, K, Traits, Timed>::make_bisc(
    const block_index_space<NA> &bisa,
    const permutation<NA> &perma,
    const block_index_space<NB> &bisb,
    const permutation<NB> &permb,
    const permutation<NC> &permc) {

    static const char *method = "make_bisc()";

    //  Block index spaces and dimensions of A and B
    //  in the standard index ordering:
    //  A(ij..pq..) B(mn..pq..)

    block_index_space<NA> bisa1(bisa);
    bisa1.permute(perma);
    block_index_space<NB> bisb1(bisb);
    bisb1.permute(permb);
    dimensions<NA> dimsa1(bisa1.get_dims());
    dimensions<NB> dimsb1(bisb1.get_dims());

    //  Build the dimensions of the result

    index<NC> i1, i2;
    for(size_t i = 0; i < N; i++) i2[i] = dimsa1[i] - 1;
    for(size_t i = 0; i < M; i++) i2[N + i] = dimsb1[i] - 1;
    for(size_t i = 0; i < K; i++) {
        if(dimsa1[N + i] != dimsb1[M + i]) {
            throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bta,btb");
        }
        if(!bisa1.get_splits(bisa1.get_type(N + i)).equals(
            bisb1.get_splits(bisb1.get_type(M + i)))) {
            throw bad_block_index_space(g_ns, k_clazz, method,
                __FILE__, __LINE__, "bta,btb");
        }
        i2[N + M + i] = dimsa1[N + i] - 1;
    }
    dimensions<NC> dimsc(index_range<NC>(i1, i2));
    block_index_space<NC> bisc(dimsc);

    //  Transfer block index space splits

    mask<NC> mfin, mdone, mtodo;
    for(size_t i = 0; i < NC; i++) mfin[i] = true;
    while(!mdone.equals(mfin)) {
        size_t i;
        for(i = 0; i < NC; i++) mtodo[i] = false;
        for(i = 0; i < NC; i++) if(!mdone[NC - i - 1]) break;
        i = NC - i - 1;
        const split_points *sp = 0;
        if(i < N) {
            size_t j = i;
            size_t typa = bisa1.get_type(j);
            for(size_t k = 0; k < N; k++) {
                mtodo[k] = (bisa1.get_type(k) == typa);
            }
            sp = &bisa1.get_splits(typa);
        } else if(i < N + M) {
            size_t j = i - N;
            size_t typb = bisb1.get_type(j);
            for(size_t k = 0; k < M; k++) {
                mtodo[N + k] = (bisb1.get_type(k) == typb);
            }
            sp = &bisb1.get_splits(typb);
        } else {
            size_t j = i - N - M;
            size_t typa = bisa1.get_type(N + j);
            size_t typb = bisb1.get_type(M + j);
            for(size_t k = 0; k < N; k++) {
                mtodo[k] = (bisa1.get_type(k) == typa);
            }
            for(size_t k = 0; k < M; k++) {
                mtodo[N + k] = (bisb1.get_type(k) == typb);
            }
            for(size_t k = 0; k < K; k++) {
                bool b1 = (bisa1.get_type(N + k) == typa);
                bool b2 = (bisb1.get_type(M + k) == typb);
                if(b1 != b2) {
                    throw bad_block_index_space(g_ns,
                        k_clazz, method,
                        __FILE__, __LINE__, "bta,btb");
                }
                mtodo[N + M + k] = b1;
            }
            sp = &bisa1.get_splits(typa);
        }
        for(size_t j = 0; j < sp->get_num_points(); j++) {
            bisc.split(mtodo, (*sp)[j]);
        }
        mdone |= mtodo;
    }

    bisc.permute(permc);
    return bisc;
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2<N, M, K, Traits, Timed>::make_symc() {

    sequence<NA, size_t> seq2a;
    sequence<NB, size_t> seq2b;
    sequence<NC, size_t> seq2c;
    for (register size_t i = 0; i < NA; i++) seq2a[i] = i;
    m_tra.get_perm().apply(seq2a);
    for (register size_t i = 0, j = NA; i < NB; i++, j++)
        seq2b[i] = j;
    m_trb.get_perm().apply(seq2b);
    for (register size_t i = 0; i < NC; i++) seq2c[i] = i;
    m_trc.get_perm().apply(seq2c);

    sequence<NC, size_t> seq2imx, seqx;
    sequence<NA + NB, size_t> seq1im, seq2im, seq;
    for (register size_t i = 0; i < N; i++) {
        seq1im[i] = i;
        seq2imx[i] = seq2a[i];
    }
    for (register size_t i = 0, j = N; i < M; i++, j++) {
        seq1im[j] = j;
        seq2imx[j] = seq2b[i];
    }

    mask<NC> mskx;
    mask<NA + NB> msk;
    for (register size_t i = 0, j = N + M; i < K; i++, j++) {
        seq1im[j] = j;
        seq2imx[j] = seq2a[i + N];
        mskx[j] = true;
        seqx[j] = i;
    }

    for (register size_t i = 0, j = NC; i < K; i++, j++) {
        seq1im[j] = j;
        seq2im[j] = seq2b[i + M];
        msk[j] = true;
        seq[j] = i;
    }

    for (register size_t i = 0; i < NC; i++) {
        seq2im[i] = seq2imx[seq2c[i]];
        seq[i] = seqx[seq2c[i]];
        msk[i] = mskx[seq2c[i]];
    }

    permutation_builder<NA + NB> pb(seq2im, seq1im);

    block_index_space_product_builder<NA, NB> bbx(m_bta.get_bis(),
            m_btb.get_bis(), pb.get_perm());

    symmetry<NA + NB, element_type> symx(bbx.get_bis());

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(m_btb);
    so_dirprod<NA, NB, element_type>(ca.req_const_symmetry(),
            cb.req_const_symmetry(), pb.get_perm()).perform(symx);

    so_merge<NA + NB, K, element_type>(symx, msk, seq).perform(m_symc);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2<N, M, K, Traits, Timed>::make_schedule() {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ctrla(m_bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> ctrlb(m_btb);

    gen_bto_ewmult2::start_timer("make_schedule");

    orbit_list<NC, element_type> ol(m_symc);
    for(typename orbit_list<NC, element_type>::iterator io = ol.begin();
        io != ol.end(); io++) {

        index<NA> bidxa;
        index<NB> bidxb;
        index<NC> bidxstd(ol.get_index(io));
        bidxstd.permute(permutation<NC>(m_trc.get_perm(), true));
        for(size_t i = 0; i < N; i++) bidxa[i] = bidxstd[i];
        for(size_t i = 0; i < M; i++) bidxb[i] = bidxstd[N + i];
        for(size_t i = 0; i < K; i++) {
            bidxa[N + i] = bidxb[M + i] = bidxstd[N + M +i];
        }
        bidxa.permute(permutation<NA>(m_tra.get_perm(), true));
        bidxb.permute(permutation<NB>(m_trb.get_perm(), true));

        orbit<NA, element_type> oa(ctrla.req_const_symmetry(), bidxa);
        orbit<NB, element_type> ob(ctrlb.req_const_symmetry(), bidxb);
        if(!oa.is_allowed() || !ob.is_allowed()) continue;

        index<NA> idxa0;
        abs_index<NA>::get_index(oa.get_abs_canonical_index(),
            m_bta.get_bis().get_block_index_dims(), idxa0);
        index<NB> idxb0;
        abs_index<NB>::get_index(ob.get_abs_canonical_index(),
            m_btb.get_bis().get_block_index_dims(), idxb0);
        bool zeroa = ctrla.req_is_zero_block(idxa0);
        bool zerob = ctrlb.req_is_zero_block(idxb0);
        if(zeroa || zerob) continue;

        m_sch.insert(ol.get_abs_index(io));
    }

    gen_bto_ewmult2::stop_timer("make_schedule");
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_ewmult2_task<N, M, K, Traits, Timed>::gen_bto_ewmult2_task(
    gen_bto_ewmult2<N, M, K, Traits, Timed> &bto,
    gen_block_tensor_i<N + M + K, bti_traits> &btc,
    const index<N + M + K> &idx,
    gen_block_stream_i<N + M + K, bti_traits> &out) :

    m_gbto(bto), m_btc(btc), m_idx(idx), m_out(out) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
void gen_bto_ewmult2_task<N, M, K, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<N + M + K>::type
        rd_block_type;
    typedef typename bti_traits::template wr_block_type<N + M + K>::type
        wr_block_type;

    tensor_transf<N + M + K, element_type> tr0;
    gen_block_tensor_ctrl<N + M + K, bti_traits> cc(m_btc);

    {
        wr_block_type &blk = cc.req_block(m_idx);
        m_gbto.compute_block_untimed(true, m_idx, tr0, blk);
        cc.ret_block(m_idx);
    }
    {
        rd_block_type &blk = cc.req_const_block(m_idx);
        m_out.put(m_idx, blk, tr0);
        cc.ret_const_block(m_idx);
    }
    cc.req_zero_block(m_idx);
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
gen_bto_ewmult2_task_iterator<N, M, K, Traits, Timed>::gen_bto_ewmult2_task_iterator(
    gen_bto_ewmult2<N, M, K, Traits, Timed> &bto,
    gen_block_tensor_i<N + M + K, bti_traits> &btc,
    gen_block_stream_i<N + M + K, bti_traits> &out) :

    m_bto(bto), m_btc(btc), m_out(out), m_sch(m_bto.get_schedule()),
    m_i(m_sch.begin()) {

}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
bool gen_bto_ewmult2_task_iterator<N, M, K, Traits, Timed>::has_more() const {

    return m_i != m_sch.end();
}


template<size_t N, size_t M, size_t K, typename Traits, typename Timed>
libutil::task_i *
gen_bto_ewmult2_task_iterator<N, M, K, Traits, Timed>::get_next() {

    dimensions<N + M + K> bidims = m_btc.get_bis().get_block_index_dims();
    index<N + M + K> idx;
    abs_index<N + M + K>::get_index(m_sch.get_abs_index(m_i), bidims, idx);
    gen_bto_ewmult2_task<N, M, K, Traits, Timed> *t =
        new gen_bto_ewmult2_task<N, M, K, Traits, Timed>(m_bto,
                m_btc, idx, m_out);
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K>
void gen_bto_ewmult2_task_observer<N, M, K>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EWMULT2_IMPL_H
