#ifndef LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
#define LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H

#include <list>
#include <set>
#include <utility>
#include <vector>
#include <libutil/threads/auto_lock.h>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include "../bto_contract2_nzorb.h"

namespace libtensor {


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task : public libutil::task_i {
private:
    const contraction2<N, M, K> &m_contr;
    block_tensor_i<N + K, T> &m_bta;
    block_tensor_i<M + K, T> &m_btb;
    const dimensions<N + K> &m_bidimsa;
    const dimensions<M + K> &m_bidimsb;
    const dimensions<N + M> &m_bidimsc;
    index<N + M> m_ic;
    std::vector<size_t> &m_blst;
    libutil::mutex &m_mtx;

public:
    bto_contract2_nzorb_task(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const dimensions<N + K> &bidimsa,
        const dimensions<M + K> &bidimsb,
        const dimensions<N + M> &bidimsc,
        const index<N + M> &ic,
        std::vector<size_t> &blst,
        libutil::mutex &mtx);
    virtual ~bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, typename T>
class bto_contract2_nzorb_task<N, M, 0, T> : public libutil::task_i {
private:
    const contraction2<N, M, 0> &m_contr;
    block_tensor_i<N, T> &m_bta;
    block_tensor_i<M, T> &m_btb;
    const dimensions<N> &m_bidimsa;
    const dimensions<M> &m_bidimsb;
    const dimensions<N + M> &m_bidimsc;
    index<N + M> m_ic;
    std::vector<size_t> &m_blst;
    libutil::mutex &m_mtx;

public:
    bto_contract2_nzorb_task(
        const contraction2<N, M, 0> &contr,
        block_tensor_i<N, T> &bta,
        block_tensor_i<M, T> &btb,
        const dimensions<N> &bidimsa,
        const dimensions<M> &bidimsb,
        const dimensions<N + M> &bidimsc,
        const index<N + M> &ic,
        std::vector<size_t> &blst,
        libutil::mutex &mtx);
    virtual ~bto_contract2_nzorb_task() { }
    virtual void perform();

};


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task_iterator : public libutil::task_iterator_i {
private:
    const contraction2<N, M, K> &m_contr;
    block_tensor_i<N + K, T> &m_bta;
    block_tensor_i<M + K, T> &m_btb;
    const symmetry<N + M, T> &m_symc;
    dimensions<N + K> m_bidimsa;
    dimensions<M + K> m_bidimsb;
    dimensions<N + M> m_bidimsc;
    orbit_list<N + M, T> m_olc;
    typename orbit_list<N + M, T>::iterator m_ioc;
    std::vector<size_t> &m_blst;
    libutil::mutex m_mtx;

public:
    bto_contract2_nzorb_task_iterator(
        const contraction2<N, M, K> &contr,
        block_tensor_i<N + K, T> &bta,
        block_tensor_i<M + K, T> &btb,
        const symmetry<N + M, T> &symc,
        std::vector<size_t> &blst);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, size_t M, size_t K, typename T>
class bto_contract2_nzorb_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


template<size_t N, size_t M, size_t K, typename T>
const char *bto_contract2_nzorb<N, M, K, T>::k_clazz =
    "bto_contract2_nzorb<N, M, K, T>";


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb<N, M, K, T>::bto_contract2_nzorb(
    const contraction2<N, M, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<M + K, T> &btb, const symmetry<N + M, T> &symc) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc.get_bis()) {

    so_copy<N + M, T>(symc).perform(m_symc);
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb<N, M, K, T>::build() {

    bto_contract2_nzorb<N, M, K, T>::start_timer();

    try {

        block_tensor_ctrl<N + K, T> ca(m_bta);
        block_tensor_ctrl<M + K, T> cb(m_btb);

        ca.req_sync_on();
        cb.req_sync_on();

        bto_contract2_nzorb_task_iterator<N, M, K, T> ti(m_contr, m_bta, m_btb,
            m_symc, m_blst);
        bto_contract2_nzorb_task_observer<N, M, K, T> to;
        libutil::thread_pool::submit(ti, to);

        ca.req_sync_off();
        cb.req_sync_off();

    } catch(...) {
        bto_contract2_nzorb<N, M, K, T>::stop_timer();
        throw;
    }

    bto_contract2_nzorb<N, M, K, T>::stop_timer();
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task<N, M, K, T>::bto_contract2_nzorb_task(
    const contraction2<N, M, K> &contr, block_tensor_i<N + K, T> &bta,
    block_tensor_i<M + K, T> &btb, const dimensions<N + K> &bidimsa,
    const dimensions<M + K> &bidimsb, const dimensions<N + M> &bidimsc,
    const index<N + M> &ic, std::vector<size_t> &blst, libutil::mutex &mtx) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_bidimsa(bidimsa),
    m_bidimsb(bidimsb), m_bidimsc(bidimsc), m_ic(ic), m_blst(blst), m_mtx(mtx) {

}


template<size_t N, size_t M, typename T>
bto_contract2_nzorb_task<N, M, 0, T>::bto_contract2_nzorb_task(
    const contraction2<N, M, 0> &contr, block_tensor_i<N, T> &bta,
    block_tensor_i<M, T> &btb, const dimensions<N> &bidimsa,
    const dimensions<M> &bidimsb, const dimensions<N + M> &bidimsc,
    const index<N + M> &ic, std::vector<size_t> &blst, libutil::mutex &mtx) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_bidimsa(bidimsa),
    m_bidimsb(bidimsb), m_bidimsc(bidimsc), m_ic(ic), m_blst(blst), m_mtx(mtx) {

}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task<N, M, K, T>::perform() {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.
    //  (This is a shortened version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    typedef std::pair< tensor_transf<N + K, T>, tensor_transf<M + K, T> >
        tensor_transf_pair_type;
    typedef std::list<tensor_transf_pair_type> contr_list_type;

    block_tensor_ctrl<N + K, T> ca(m_bta);
    block_tensor_ctrl<M + K, T> cb(m_btb);

    const sequence<2 * (N + M + K), size_t> &conn = m_contr.get_conn();
    const symmetry<N + K, T> &syma = ca.req_const_symmetry();
    const symmetry<M + K, T> &symb = cb.req_const_symmetry();

    index<K> ik1, ik2;
    for(size_t i = 0, j = 0; i < N + K; i++) {
        if(conn[N + M + i] > N + M) {
            ik2[j++] = m_bidimsa[i] - 1;
        }
    }
    dimensions<K> bidimsk(index_range<K>(ik1, ik2));
    std::set<size_t> ikset;
    size_t nk = bidimsk.get_size();
    for(size_t i = 0; i < nk; i++) ikset.insert(i);

    while(!ikset.empty()) {

        index<N + K> ia;
        index<M + K> ib;
        const index<N + M> &ic = m_ic;
        index<K> ik;
        abs_index<K>::get_index(*ikset.begin(), bidimsk, ik);
        sequence<K, size_t> ka(0), kb(0);

        //  Determine ia, ib from ic, ik
        for(size_t i = 0, j = 0; i < N + K; i++) {
            if(conn[N + M + i] < N + M) {
                ia[i] = ic[conn[N + M + i]];
            } else {
                ka[j] = i;
                kb[j] = conn[N + M + i] - 2 * N - M - K;
                ia[ka[j]] = ib[kb[j]] = ik[j];
                j++;
            }
        }
        for(size_t i = 0; i < M + K; i++) {
            if(conn[2 * N + M + K + i] < N + M) {
                ib[i] = ic[conn[2 * N + M + K + i]];
            }
        }

        orbit<N + K, T> oa(syma, ia);
        orbit<M + K, T> ob(symb, ib);
        bool zero = !oa.is_allowed() || !ob.is_allowed();

        abs_index<N + K> acia(oa.get_abs_canonical_index(), m_bidimsa);
        abs_index<M + K> acib(ob.get_abs_canonical_index(), m_bidimsb);
        if(!zero) {
            zero = ca.req_is_zero_block(acia.get_index()) ||
                cb.req_is_zero_block(acib.get_index());
        }

        contr_list_type clist;

        //  Build the list of contractions for the current orbits A, B

        typename orbit<N + K, T>::iterator ja;
        typename orbit<M + K, T>::iterator jb;
        for(ja = oa.begin(); ja != oa.end(); ++ja)
        for(jb = ob.begin(); jb != ob.end(); ++jb) {
            index<N + K> ia1;
            index<M + K> ib1;
            abs_index<N + K>::get_index(oa.get_abs_index(ja), m_bidimsa, ia1);
            abs_index<M + K>::get_index(ob.get_abs_index(jb), m_bidimsb, ib1);
            index<N + M> ic1;
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
                clist.push_back(tensor_transf_pair_type(oa.get_transf(ja),
                    ob.get_transf(jb)));
            }
            ikset.erase(abs_index<K>::get_abs_index(ika, bidimsk));
        }

        //  Coalesce contractions if possible

        typename contr_list_type::iterator j1 = clist.begin();
        while(j1 != clist.end()) {
            typename contr_list_type::iterator j2 = j1;
            ++j2;
            bool incj1 = true;
            while(j2 != clist.end()) {
                if(j1->first.get_perm().equals(j2->first.get_perm()) &&
                    j1->second.get_perm().equals(j2->second.get_perm())) {
                    // TODO: replace with scalar_transf::combine
                    // TODO: this code is double-specific!
                    double d1 = j1->first.get_scalar_tr().get_coeff() *
                        j1->second.get_scalar_tr().get_coeff();
                    double d2 = j2->first.get_scalar_tr().get_coeff() *
                        j2->second.get_scalar_tr().get_coeff();
                    if(d1 + d2 == 0) {
                        j1 = clist.erase(j1);
                        if(j1 == j2) {
                            j1 = j2 = clist.erase(j2);
                        } else {
                            j2 = clist.erase(j2);
                        }
                        incj1 = false;
                        break;
                    } else {
                        j1->first.get_scalar_tr().reset();
                        j1->second.get_scalar_tr().reset();
                        j1->first.get_scalar_tr().scale(d1 + d2);
                        j2 = clist.erase(j2);
                    }
                } else {
                    ++j2;
                }
            }
            if(incj1) ++j1;
        }

        //  If the list is not empty, there is no need to continue
        //  the block is non-zero

        if(!clist.empty()) {
            libutil::auto_lock<libutil::mutex> lock(m_mtx);
            m_blst.push_back(abs_index<N + M>::get_abs_index(m_ic, m_bidimsc));
            return;
        }
    }
}


template<size_t N, size_t M, typename T>
void bto_contract2_nzorb_task<N, M, 0, T>::perform() {

    //  For a specified block in the result block tensor (C),
    //  this algorithm makes a list of contractions of the blocks
    //  from A and B that need to be made.
    //  (This is a shortened version of the algorithm, which terminates
    //  as soon as it is clear that at least one contraction is required
    //  and therefore the block in C is non-zero.)

    typedef std::pair< tensor_transf<N, T>, tensor_transf<M, T> >
        tensor_transf_pair_type;
    typedef std::list<tensor_transf_pair_type> contr_list_type;

    block_tensor_ctrl<N, T> ca(m_bta);
    block_tensor_ctrl<M, T> cb(m_btb);

    const sequence<2 * (N + M), size_t> &conn = m_contr.get_conn();
    const symmetry<N, T> &syma = ca.req_const_symmetry();
    const symmetry<M, T> &symb = cb.req_const_symmetry();

    index<N> ia;
    index<M> ib;
    const index<N + M> &ic = m_ic;

    //  Determine ia, ib from ic
    for(size_t i = 0, j = 0; i < N; i++) {
        ia[i] = ic[conn[N + M + i]];
    }
    for(size_t i = 0; i < M; i++) {
        ib[i] = ic[conn[2 * N + M + i]];
    }

    orbit<N, T> oa(syma, ia);
    orbit<M, T> ob(symb, ib);

    abs_index<N> acia(oa.get_abs_canonical_index(), m_bidimsa);
    abs_index<M> acib(ob.get_abs_canonical_index(), m_bidimsb);
    bool zero = ca.req_is_zero_block(acia.get_index()) ||
        cb.req_is_zero_block(acib.get_index());

    contr_list_type clist;

    //  Build the list of contractions for the current orbits A, B

    {
        typename orbit<N, T>::iterator ja;
        typename orbit<M, T>::iterator jb;
        for(ja = oa.begin(); ja != oa.end(); ++ja)
        for(jb = ob.begin(); jb != ob.end(); ++jb) {
            index<N> ia1;
            index<M> ib1;
            abs_index<N>::get_index(oa.get_abs_index(ja), m_bidimsa, ia1);
            abs_index<M>::get_index(ob.get_abs_index(jb), m_bidimsb, ib1);
            index<N + M> ic1;
            for(size_t i = 0; i < N + M; i++) {
                if(conn[i] >= 2 * N + M) {
                    ic1[i] = ib1[conn[i] - 2 * N - M];
                } else {
                    ic1[i] = ia1[conn[i] - N - M];
                }
            }
            if(!ic1.equals(ic)) continue;
            clist.push_back(tensor_transf_pair_type(oa.get_transf(ja),
                ob.get_transf(jb)));
        }
    }

    //  Coalesce contractions if possible

    typename contr_list_type::iterator j1 = clist.begin();
    while(j1 != clist.end()) {
        typename contr_list_type::iterator j2 = j1;
        ++j2;
        bool incj1 = true;
        while(j2 != clist.end()) {
            if(j1->first.get_perm().equals(j2->first.get_perm()) &&
                j1->second.get_perm().equals(j2->second.get_perm())) {
                // TODO: replace with scalar_transf::combine
                // TODO: this code is double-specific!
                double d1 = j1->first.get_scalar_tr().get_coeff() *
                    j1->second.get_scalar_tr().get_coeff();
                double d2 = j2->first.get_scalar_tr().get_coeff() *
                    j2->second.get_scalar_tr().get_coeff();
                if(d1 + d2 == 0) {
                    j1 = clist.erase(j1);
                    if(j1 == j2) {
                        j1 = j2 = clist.erase(j2);
                    } else {
                        j2 = clist.erase(j2);
                    }
                    incj1 = false;
                    break;
                } else {
                    j1->first.get_scalar_tr().reset();
                    j1->second.get_scalar_tr().reset();
                    j1->first.get_scalar_tr().scale(d1 + d2);
                    j2 = clist.erase(j2);
                }
            } else {
                ++j2;
            }
        }
        if(incj1) ++j1;
    }

    //  If the list is not empty, there is no need to continue
    //  the block is non-zero

    if(!clist.empty()) {
        libutil::auto_lock<libutil::mutex> lock(m_mtx);
        m_blst.push_back(abs_index<N + M>::get_abs_index(m_ic, m_bidimsc));
        return;
    }
}


template<size_t N, size_t M, size_t K, typename T>
bto_contract2_nzorb_task_iterator<N, M, K, T>::
    bto_contract2_nzorb_task_iterator(const contraction2<N, M, K> &contr,
    block_tensor_i<N + K, T> &bta, block_tensor_i<M + K, T> &btb,
    const symmetry<N + M, T> &symc, std::vector<size_t> &blst) :

    m_contr(contr), m_bta(bta), m_btb(btb), m_symc(symc),
    m_bidimsa(m_bta.get_bis().get_block_index_dims()),
    m_bidimsb(m_btb.get_bis().get_block_index_dims()),
    m_bidimsc(m_symc.get_bis().get_block_index_dims()),
    m_olc(m_symc), m_ioc(m_olc.begin()), m_blst(blst) {

}


template<size_t N, size_t M, size_t K, typename T>
bool bto_contract2_nzorb_task_iterator<N, M, K, T>::has_more() const {

    return m_ioc != m_olc.end();
}


template<size_t N, size_t M, size_t K, typename T>
libutil::task_i *bto_contract2_nzorb_task_iterator<N, M, K, T>::get_next() {

    bto_contract2_nzorb_task<N, M, K, T> *t =
        new bto_contract2_nzorb_task<N, M, K, T>(m_contr, m_bta, m_btb,
            m_bidimsa, m_bidimsb, m_bidimsc, m_olc.get_index(m_ioc),
            m_blst, m_mtx);
    ++m_ioc;
    return t;
}


template<size_t N, size_t M, size_t K, typename T>
void bto_contract2_nzorb_task_observer<N, M, K, T>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTO_CONTRACT2_NZORB_IMPL_H
