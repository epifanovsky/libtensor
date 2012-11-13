#ifndef LIBTENSOR_GEN_BTO_TRACE_IMPL_H
#define LIBTENSOR_GEN_BTO_TRACE_IMPL_H

#include <vector>
#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include "../gen_bto_trace.h"

namespace libtensor {


template<size_t N, typename Traits, typename Timed>
class gen_bto_trace_in_orbit_task :
    public libutil::task_i, public timings<Timed> {

    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

public:
    static const char *k_clazz;

    enum {
        NA = 2 * N
    };
private:
    gen_block_tensor_rd_i<NA, bti_traits> &m_bta;
    const permutation<NA> &m_perm;
    const orbit_list<NA, element_type> &m_ola;
    const dimensions<NA> &m_bidimsa;
    index<NA> m_idxa;
    element_type m_tr;

public:
    gen_bto_trace_in_orbit_task(
            gen_block_tensor_rd_i<NA, bti_traits> &bta,
            const permutation<NA> &perm,
            const orbit_list<NA, element_type> &ola,
            const index<NA> &idxa,
            const dimensions<NA> &bidimsa) :
        m_bta(bta), m_ola(ola), m_bidimsa(bidimsa), m_idxa(idxa),
        m_perm(perm), m_tr(Traits::zero()) { }

    virtual ~gen_bto_trace_in_orbit_task() { }
    virtual void perform();

    element_type get_trace() const { return m_tr; }
};


template<size_t N, typename Traits, typename Timed>
class gen_bto_trace_task_iterator : public libutil::task_iterator_i {
public:
    typedef gen_bto_trace_in_orbit_task<N, Traits, Timed> task_type;

private:
    std::vector<task_type *> &m_tl;
    typename std::vector<task_type *>::iterator m_i;

public:
    gen_bto_trace_task_iterator(std::vector<task_type *> &tl) :
        m_tl(tl), m_i(m_tl.begin()) {

    }

    virtual ~gen_bto_trace_task_iterator() { }

    virtual bool has_more() const;

    virtual libutil::task_i *get_next();
};


template<size_t N, typename Traits>
class gen_bto_trace_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t) { }
};


template<size_t N, typename Traits, typename Timed>
const char *gen_bto_trace<N, Traits, Timed>::k_clazz =
        "gen_bto_trace<N, Traits, Timed>";


template<size_t N, typename Traits, typename Timed>
gen_bto_trace<N, Traits, Timed>::gen_bto_trace(
        gen_block_tensor_rd_i<NA, bti_traits> &bta,
        const permutation<NA> &perm) :
        m_bta(bta), m_perm(perm) {

}


template<size_t N, typename Traits, typename Timed>
typename Traits::element_type gen_bto_trace<N, Traits, Timed>::calculate() {

    typedef gen_bto_trace_in_orbit_task<N, Traits, Timed> task_type;

    gen_bto_trace::start_timer();

    element_type tr = Traits::zero();

    try {

    dimensions<NA> bidimsa = m_bta.get_bis().get_block_index_dims();

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);

    std::vector<task_type *> tasklist;

    orbit_list<NA, element_type> ola(ca.req_const_symmetry());
    for (typename orbit_list<NA, element_type>::iterator ioa = ola.begin();
            ioa != ola.end(); ioa++) {

        index<NA> idxa;
        ola.get_index(ioa, idxa);
        if(ca.req_is_zero_block(idxa)) continue;

        task_type *t = new task_type(m_bta, m_perm, ola, idxa, bidimsa);
        tasklist.push_back(t);
    }

    gen_bto_trace_task_iterator<N, Traits, Timed> ti(tasklist);
    gen_bto_trace_task_observer<N, Traits> to;
    libutil::thread_pool::submit(ti, to);

    for(size_t k = 0; k < tasklist.size(); k++) {
        tr += tasklist[k]->get_trace();
        delete tasklist[k];
    }


    } catch(...) {
        gen_bto_trace::stop_timer();
        throw;
    }

    gen_bto_trace::stop_timer();

    return tr;
}


template<size_t N, typename Traits, typename Timed>
void gen_bto_trace_in_orbit_task<N, Traits, Timed>::perform() {

    typedef typename bti_traits::template rd_block_type<NA>::type rd_block_type;
    typedef typename Traits::template to_trace_type<N>::type to_trace_type;

    gen_block_tensor_rd_ctrl<NA, bti_traits> ca(m_bta);

    gen_bto_trace_in_orbit_task::start_timer();

    try {

        rd_block_type *ba = 0;

        orbit<NA, element_type> oa(ca.req_const_symmetry(), m_idxa);
        for (typename orbit<NA, element_type>::iterator ioa = oa.begin();
                ioa != oa.end(); ioa++) {

            index<NA> ia;
            abs_index<NA>::get_index(oa.get_abs_index(ioa), m_bidimsa, ia);
            ia.permute(m_perm);

            bool skip = false;
            for(register size_t i = 0; i < N; i++) {
                if(ia[i] != ia[N + i]) {
                    skip = true;
                    break;
                }
            }
            if(skip) continue;

            tensor_transf<NA, element_type> tra(oa.get_transf(ioa));
            tra.permute(m_perm);

            if (ba == 0) ba = &ca.req_const_block(oa.get_cindex());
            element_type tr0 = to_trace_type(*ba, tra.get_perm()).calculate();
            tra.get_scalar_tr().apply(tr0);
            m_tr += tr0;
        }
        if(ba != 0) ca.ret_const_block(oa.get_cindex());

    } catch (...) {
        gen_bto_trace_in_orbit_task::stop_timer();
        throw;
    }

    gen_bto_trace_in_orbit_task::stop_timer();
}


template<size_t N, typename Traits, typename Timed>
bool gen_bto_trace_task_iterator<N, Traits, Timed>::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, typename Traits, typename Timed>
libutil::task_i *gen_bto_trace_task_iterator<N, Traits, Timed>::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_TRACE_IMPL_H
