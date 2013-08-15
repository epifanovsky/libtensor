#ifndef LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H
#define LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H

#include <list>
#include <set>
#include <utility>
#include <vector>
#include <libutil/threads/auto_lock.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/core/short_orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../gen_block_tensor_ctrl.h"
#include "gen_bto_contract2_block_list.h"
#include "gen_bto_contract2_clst_builder.h"
#include "gen_bto_contract2_nzorb.h"
#include "gen_bto_unfold_block_list.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename Traits>
struct gen_bto_contract2_nzorb_task_ctx {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

public:
    const contraction2<N, M, K> &m_contr;
    const symmetry<NA, element_type> &m_syma;
    const symmetry<NB, element_type> &m_symb;
    const symmetry<NC, element_type> &m_symc;
    dimensions<NA> m_bidimsa;
    dimensions<NB> m_bidimsb;
    dimensions<NC> m_bidimsc;
    const block_list<NA> &m_blsta;
    const block_list<NB> &m_blstb;
    const gen_bto_contract2_block_list<N, M, K> &m_cbl;
    std::vector<size_t> &m_visited;
    std::vector<size_t> &m_nonzero;
    libutil::mutex &m_vis_mtx;
    libutil::mutex &m_nz_mtx;

public:
    gen_bto_contract2_nzorb_task_ctx(
        const contraction2<N, M, K> &contr,
        const symmetry<NA, element_type> &syma,
        const symmetry<NB, element_type> &symb,
        const symmetry<NC, element_type> &symc,
        const block_list<NA> &blsta,
        const block_list<NB> &blstb,
        const gen_bto_contract2_block_list<N, M, K> &cbl,
        std::vector<size_t> &visited,
        std::vector<size_t> &nonzero,
        libutil::mutex &vis_mtx,
        libutil::mutex &nz_mtx) :

        m_contr(contr), m_syma(syma), m_symb(symb), m_symc(symc),
        m_bidimsa(syma.get_bis().get_block_index_dims()),
        m_bidimsb(symb.get_bis().get_block_index_dims()),
        m_bidimsc(symc.get_bis().get_block_index_dims()),
        m_blsta(blsta), m_blstb(blstb), m_cbl(cbl),
        m_visited(visited), m_nonzero(nonzero),
        m_vis_mtx(vis_mtx), m_nz_mtx(nz_mtx)
    { }

};


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_nzorb_task : public libutil::task_i {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> &m_ctx;
    size_t m_k;

public:
    gen_bto_contract2_nzorb_task(
        gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> &ctx,
        size_t k) :
        m_ctx(ctx), m_k(k)
    { }

    virtual ~gen_bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, typename Traits>
class gen_bto_contract2_nzorb_task<N, M, 0, Traits> : public libutil::task_i {
public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_nzorb_task_ctx<N, M, 0, Traits> &m_ctx;
    size_t m_i;

public:
    gen_bto_contract2_nzorb_task(
        gen_bto_contract2_nzorb_task_ctx<N, M, 0, Traits> &ctx,
        size_t i) :
        m_ctx(ctx), m_i(i)
    { }

    virtual ~gen_bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename Traits>
class gen_bto_contract2_nzorb_task_iterator : public libutil::task_iterator_i {
public:
    enum {
        NA = N + K, //!< Order of first argument (A)
        NB = M + K, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> &m_ctx;
    std::vector<size_t> m_k;
    typename std::vector<size_t>::const_iterator m_i;

public:
    gen_bto_contract2_nzorb_task_iterator(
        gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> &ctx);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, typename Traits>
class gen_bto_contract2_nzorb_task_iterator<N, M, 0, Traits> :
    public libutil::task_iterator_i {

public:
    enum {
        NA = N, //!< Order of first argument (A)
        NB = M, //!< Order of second argument (B)
        NC = N + M  //!< Order of result (C)
    };

public:
    //! Type of tensor elements
    typedef typename Traits::element_type element_type;

    //! Block tensor interface traits
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_bto_contract2_nzorb_task_ctx<N, M, 0, Traits> &m_ctx;
    typename block_list<NA>::iterator m_i;

public:
    gen_bto_contract2_nzorb_task_iterator(
        gen_bto_contract2_nzorb_task_ctx<N, M, 0, Traits> &ctx);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K>
class gen_bto_contract2_nzorb_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb<N, M, K, Traits>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(bta.get_bis()), m_symb(btb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(bta.get_bis().get_block_index_dims()),
    m_blstb(btb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);
    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    so_copy<NA, element_type>(ca.req_const_symmetry()).perform(m_syma);
    so_copy<NB, element_type>(cb.req_const_symmetry()).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    std::vector<size_t> blst;
    ca.req_nonzero_blocks(blst);
    for(size_t i = 0; i < blst.size(); i++) m_blsta.add(blst[i]);
    cb.req_nonzero_blocks(blst);
    for(size_t i = 0; i < blst.size(); i++) m_blstb.add(blst[i]);
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb<N, M, K, Traits>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const assignment_schedule<NA, element_type> &scha,
    gen_block_tensor_rd_i<NB, bti_traits> &btb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(syma.get_bis()), m_symb(btb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(syma.get_bis().get_block_index_dims()),
    m_blstb(btb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NB, bti_traits> cb(btb);

    so_copy<NA, element_type>(syma).perform(m_syma);
    so_copy<NB, element_type>(cb.req_const_symmetry()).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    for(typename assignment_schedule<NA, element_type>::iterator ia =
        scha.begin(); ia != scha.end(); ++ia) {
        m_blsta.add(scha.get_abs_index(ia));
    }

    orbit_list<NB, element_type> olb(m_symb);
    for(typename orbit_list<NB, element_type>::iterator iol = olb.begin();
        iol != olb.end(); ++iol) {
        index<NB> idx;
        olb.get_index(iol, idx);
        if(cb.req_is_zero_block(idx)) continue;
        m_blstb.add(olb.get_abs_index(iol));
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb<N, M, K, Traits>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    gen_block_tensor_rd_i<NA, bti_traits> &bta,
    const symmetry<NB, element_type> &symb,
    const assignment_schedule<NB, element_type> &schb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(bta.get_bis()), m_symb(symb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(bta.get_bis().get_block_index_dims()),
    m_blstb(symb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(bta);

    so_copy<NA, element_type>(ca.req_const_symmetry()).perform(m_syma);
    so_copy<NB, element_type>(symb).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    orbit_list<NA, element_type> ola(m_syma);
    for(typename orbit_list<NA, element_type>::iterator iol = ola.begin();
        iol != ola.end(); ++iol) {
        index<NA> idx;
        ola.get_index(iol, idx);
        if(ca.req_is_zero_block(idx)) continue;
        m_blsta.add(ola.get_abs_index(iol));
    }

    for(typename assignment_schedule<NB, element_type>::iterator isch =
        schb.begin(); isch != schb.end(); ++isch) {
        m_blstb.add(schb.get_abs_index(isch));
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb<N, M, K, Traits>::gen_bto_contract2_nzorb(
    const contraction2<N, M, K> &contr,
    const symmetry<NA, element_type> &syma,
    const assignment_schedule<NA, element_type> &scha,
    const symmetry<NB, element_type> &symb,
    const assignment_schedule<NB, element_type> &schb,
    const symmetry<NC, element_type> &symc) :

    m_contr(contr),
    m_syma(syma.get_bis()), m_symb(symb.get_bis()), m_symc(symc.get_bis()),
    m_blsta(syma.get_bis().get_block_index_dims()),
    m_blstb(symb.get_bis().get_block_index_dims()),
    m_blstc(symc.get_bis().get_block_index_dims()) {

    so_copy<NA, element_type>(syma).perform(m_syma);
    so_copy<NB, element_type>(symb).perform(m_symb);
    so_copy<NC, element_type>(symc).perform(m_symc);

    for(typename assignment_schedule<NA, element_type>::iterator isch =
        scha.begin(); isch != scha.end(); ++isch) {
        m_blsta.add(scha.get_abs_index(isch));
    }

    for(typename assignment_schedule<NB, element_type>::iterator isch =
        schb.begin(); isch != schb.end(); ++isch) {
        m_blstb.add(schb.get_abs_index(isch));
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_nzorb<N, M, K, Traits>::build() {

    dimensions<NA> bidimsa = m_syma.get_bis().get_block_index_dims();
    dimensions<NB> bidimsb = m_symb.get_bis().get_block_index_dims();

    block_list<NA> blstax(bidimsa);
    block_list<NB> blstbx(bidimsb);

    gen_bto_unfold_block_list<NA, Traits>(m_syma, m_blsta).build(blstax);
    gen_bto_unfold_block_list<NB, Traits>(m_symb, m_blstb).build(blstbx);
    gen_bto_contract2_block_list<N, M, K> cbl(m_contr, bidimsa, blstax,
        bidimsb, blstbx);

    std::vector<size_t> blstc, vis;
    libutil::mutex blstc_mtx, vis_mtx;
    gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> tctx(m_contr,
        m_syma, m_symb, m_symc, blstax, blstbx, cbl, vis, blstc, vis_mtx,
        blstc_mtx);

    gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits> ti(tctx);
    gen_bto_contract2_nzorb_task_observer<N, M, K> to;
    libutil::thread_pool::submit(ti, to);

    for(size_t i = 0; i < blstc.size(); i++) m_blstc.add(blstc[i]);
}


template<size_t N, size_t M, size_t K, typename Traits>
void gen_bto_contract2_nzorb_task<N, M, K, Traits>::perform() {

    const sequence<NA + NB + NC, size_t> &conn = m_ctx.m_contr.get_conn();
    sequence<NC, size_t> seq1(0), seq2(0);
    index<NC> i1, i2, i3;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] < NC) {
            i2[j] = m_ctx.m_bidimsa[i] - 1;
            seq1[j] = NC + i;
            seq2[j] = conn[j];
            j++;
        }
    }
    for(size_t i = 0, j = 0; i < NB; i++) {
        if(conn[NC + NA + i] < NC) {
            i3[N + j] = m_ctx.m_bidimsb[i] - 1;
            seq1[N + j] = NC + NA + i;
            seq2[N + j] = conn[N + j];
            j++;
        }
    }
    dimensions<NC> dimsci(index_range<NC>(i1, i2));
    dimensions<NC> dimscj(index_range<NC>(i1, i3));
    permutation_builder<NC> pbc(seq2, seq1);
    permutation<NC> permc(pbc.get_perm());

    const std::vector< index<2> > &bla = m_ctx.m_cbl.get_blsta_1();
    const std::vector< index<2> > &blb = m_ctx.m_cbl.get_blstb_1();

    std::vector<size_t> candidates, nonzero;

    gen_bto_contract2_block_list_less_1 comp_1;
    index<2> isrch2; isrch2[0] = m_k;
    typename std::vector< index<2> >::const_iterator ia =
        std::lower_bound(bla.begin(), bla.end(), isrch2, comp_1);
    typename std::vector< index<2> >::const_iterator ib =
        std::lower_bound(blb.begin(), blb.end(), isrch2, comp_1);
    index<NC> ici, icj, ic;
    while(ia != bla.end() && ia->at(0) == m_k) {
        abs_index<NC>::get_index(ia->at(1), dimsci, ici);
        typename std::vector< index<2> >::const_iterator ib1 = ib;
        while(ib1 != blb.end() && ib1->at(0) == m_k) {
            abs_index<NC>::get_index(ib1->at(1), dimscj, icj);
            for(size_t i = 0; i < NC; i++) ic[i] = ici[i] + icj[i];
            ic.permute(permc);
            size_t aic = abs_index<NC>::get_abs_index(ic, m_ctx.m_bidimsc);
            short_orbit<NC, element_type> soc(m_ctx.m_symc, aic, true);
            if(soc.is_allowed() && soc.get_acindex() == aic) {
                candidates.push_back(aic);
            }
            ++ib1;
        }
        ++ia;
    }
    std::sort(candidates.begin(), candidates.end());

    {
        libutil::auto_lock<libutil::mutex> lock(m_ctx.m_vis_mtx);
        std::vector<size_t> tmp;
        typename std::vector<size_t>::iterator i;
        tmp.resize(candidates.size());
        i = std::set_difference(candidates.begin(), candidates.end(),
            m_ctx.m_visited.begin(), m_ctx.m_visited.end(), tmp.begin());
        tmp.resize(i - tmp.begin());
        candidates.swap(tmp);
        tmp.resize(m_ctx.m_visited.size() + candidates.size());
        i = std::merge(candidates.begin(), candidates.end(),
            m_ctx.m_visited.begin(), m_ctx.m_visited.end(), tmp.begin());
        tmp.resize(i - tmp.begin());
        m_ctx.m_visited.swap(tmp);
    }

    nonzero.reserve(candidates.size());
    for(typename std::vector<size_t>::iterator i = candidates.begin();
        i != candidates.end(); ++i) {

        index<NC> ic;
        abs_index<NC>::get_index(*i, m_ctx.m_bidimsc, ic);
        gen_bto_contract2_clst_builder<N, M, K, Traits> bld(m_ctx.m_contr,
            m_ctx.m_syma, m_ctx.m_symb, m_ctx.m_blsta, m_ctx.m_blstb,
            m_ctx.m_bidimsc, ic);
        bld.build_list(false, m_ctx.m_cbl);
        if(!bld.is_empty()) nonzero.push_back(*i);
    }

    {
        libutil::auto_lock<libutil::mutex> lock(m_ctx.m_nz_mtx);
        std::vector<size_t> nznew;
        nznew.resize(m_ctx.m_nonzero.size() + nonzero.size());
        typename std::vector<size_t>::iterator i = std::merge(
            nonzero.begin(), nonzero.end(),
            m_ctx.m_nonzero.begin(), m_ctx.m_nonzero.end(), nznew.begin());
        nznew.resize(i - nznew.begin());
        m_ctx.m_nonzero.swap(nznew);
    }
}


template<size_t N, size_t M, typename Traits>
void gen_bto_contract2_nzorb_task<N, M, 0, Traits>::perform() {

    const sequence<NA + NB + NC, size_t> &conn = m_ctx.m_contr.get_conn();
    sequence<NC, size_t> seq1(0), seq2(0);
    index<NC> i1, i2, i3;
    for(size_t i = 0, j = 0; i < NA; i++) {
        if(conn[NC + i] < NC) {
            i2[j] = m_ctx.m_bidimsa[i] - 1;
            seq1[j] = NC + i;
            seq2[j] = conn[j];
            j++;
        }
    }
    for(size_t i = 0, j = 0; i < NB; i++) {
        if(conn[NC + NA + i] < NC) {
            i3[N + j] = m_ctx.m_bidimsb[i] - 1;
            seq1[N + j] = NC + NA + i;
            seq2[N + j] = conn[N + j];
            j++;
        }
    }
    dimensions<NC> dimsci(index_range<NC>(i1, i2));
    dimensions<NC> dimscj(index_range<NC>(i1, i3));
    permutation_builder<NC> pbc(seq2, seq1);
    permutation<NC> permc(pbc.get_perm());

    std::vector<size_t> nonzero;

    index<NC> ici, icj, ic;
    abs_index<NC>::get_index(m_i, dimsci, ici);
    typename block_list<NB>::iterator ib = m_ctx.m_blstb.begin();
    while(ib != m_ctx.m_blstb.end()) {
        abs_index<NC>::get_index(m_ctx.m_blstb.get_abs_index(ib), dimscj, icj);
        for(size_t i = 0; i < NC; i++) ic[i] = ici[i] + icj[i];
        ic.permute(permc);
        size_t aic = abs_index<NC>::get_abs_index(ic, m_ctx.m_bidimsc);
        short_orbit<NC, element_type> soc(m_ctx.m_symc, aic, true);
        if(soc.is_allowed() && soc.get_acindex() == aic) nonzero.push_back(aic);
        ++ib;
    }
    std::sort(nonzero.begin(), nonzero.end());

    {
        libutil::auto_lock<libutil::mutex> lock(m_ctx.m_nz_mtx);
        std::vector<size_t> nznew;
        nznew.resize(m_ctx.m_nonzero.size() + nonzero.size());
        typename std::vector<size_t>::iterator i = std::merge(
            nonzero.begin(), nonzero.end(),
            m_ctx.m_nonzero.begin(), m_ctx.m_nonzero.end(), nznew.begin());
        nznew.resize(i - nznew.begin());
        m_ctx.m_nonzero.swap(nznew);
    }
}


template<size_t N, size_t M, size_t K, typename Traits>
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::
gen_bto_contract2_nzorb_task_iterator(
    gen_bto_contract2_nzorb_task_ctx<N, M, K, Traits> &ctx) :

    m_ctx(ctx) {

    std::vector<size_t> ka, kb;
    size_t k;

    typename std::vector< index<2> >::const_iterator ia =
        m_ctx.m_cbl.get_blsta_1().begin();
    if(ia != m_ctx.m_cbl.get_blsta_1().end()) {
        k = ia->at(0);
        ka.push_back(k);
    }
    while(ia != m_ctx.m_cbl.get_blsta_1().end()) {
        size_t k1 = ia->at(0);
        if(k1 > k) {
            k = k1;
            ka.push_back(k);
        }
        ++ia;
    }

    typename std::vector< index<2> >::const_iterator ib =
        m_ctx.m_cbl.get_blstb_1().begin();
    if(ib != m_ctx.m_cbl.get_blstb_1().end()) {
        k = ib->at(0);
        kb.push_back(k);
    }
    while(ib != m_ctx.m_cbl.get_blstb_1().end()) {
        size_t k1 = ib->at(0);
        if(k1 > k) {
            k = k1;
            kb.push_back(k);
        }
        ++ib;
    }

    m_k.resize(std::max(ka.size(), kb.size()));
    typename std::vector<size_t>::iterator kend =
        std::set_intersection(ka.begin(), ka.end(), kb.begin(), kb.end(),
            m_k.begin());
    m_k.resize(kend - m_k.begin());
    m_i = m_k.begin();
}


template<size_t N, size_t M, size_t K, typename Traits>
bool gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::has_more() const {

    return m_i != m_k.end();
}


template<size_t N, size_t M, size_t K, typename Traits>
libutil::task_i*
gen_bto_contract2_nzorb_task_iterator<N, M, K, Traits>::get_next() {

    gen_bto_contract2_nzorb_task<N, M, K, Traits> *t =
        new gen_bto_contract2_nzorb_task<N, M, K, Traits>(m_ctx, *m_i);
    ++m_i;
    return t;
}


template<size_t N, size_t M, typename Traits>
gen_bto_contract2_nzorb_task_iterator<N, M, 0, Traits>::
gen_bto_contract2_nzorb_task_iterator(
    gen_bto_contract2_nzorb_task_ctx<N, M, 0, Traits> &ctx) :

    m_ctx(ctx) {

    m_i = m_ctx.m_blsta.begin();
}


template<size_t N, size_t M, typename Traits>
bool gen_bto_contract2_nzorb_task_iterator<N, M, 0, Traits>::has_more() const {

    return m_i != m_ctx.m_blsta.end();
}


template<size_t N, size_t M, typename Traits>
libutil::task_i*
gen_bto_contract2_nzorb_task_iterator<N, M, 0, Traits>::get_next() {

    gen_bto_contract2_nzorb_task<N, M, 0, Traits> *t =
        new gen_bto_contract2_nzorb_task<N, M, 0, Traits>(m_ctx,
            m_ctx.m_blsta.get_abs_index(m_i));
    ++m_i;
    return t;
}


template<size_t N, size_t M, size_t K>
void gen_bto_contract2_nzorb_task_observer<N, M, K>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_CONTRACT2_NZORB_IMPL_H
