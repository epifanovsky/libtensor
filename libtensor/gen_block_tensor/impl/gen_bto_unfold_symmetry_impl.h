#ifndef LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_IMPL_H
#define LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_IMPL_H

#include <libutil/thread_pool/thread_pool.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/symmetry/so_copy.h>
#include "../gen_block_tensor_ctrl.h"
#include "gen_bto_unfold_symmetry.h"

namespace libtensor {


namespace {


template<size_t N, typename Traits>
class gen_bto_unfold_symmetry_task : public libutil::task_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_i<N, bti_traits> &m_bt;
    const dimensions<N> &m_bidims;
    const symmetry<N, element_type> &m_symcopy;
    size_t m_aidx;
    bool m_orbits;

public:
    gen_bto_unfold_symmetry_task(
        gen_block_tensor_i<N, bti_traits> &bta,
        const dimensions<N> &bidims,
        const symmetry<N, element_type> &symcopy,
        size_t aidx,
        bool orbits);

    virtual ~gen_bto_unfold_symmetry_task() { }
    virtual void perform();

};


template<size_t N, typename Traits>
class gen_bto_unfold_symmetry_task_iterator : public libutil::task_iterator_i {
public:
    typedef typename Traits::element_type element_type;
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_i<N, bti_traits> &m_bt;
    dimensions<N> m_bidims;
    gen_block_tensor_ctrl<N, bti_traits> m_ctrl;
    const symmetry<N, element_type> &m_symcopy;
    bool m_orbits;
    std::vector<size_t> m_nzorb;
    std::vector<size_t>::const_iterator m_i;

public:
    gen_bto_unfold_symmetry_task_iterator(
        gen_block_tensor_i<N, bti_traits> &bt,
        const symmetry<N, element_type> &symcopy);

    gen_bto_unfold_symmetry_task_iterator(
        const std::vector<size_t> &blst,
        gen_block_tensor_i<N, bti_traits> &bt,
        const symmetry<N, element_type> &symcopy);

    virtual bool has_more() const;
    virtual libutil::task_i *get_next();

};


template<size_t N, typename Traits>
class gen_bto_unfold_symmetry_task_observer : public libutil::task_observer_i {
public:
    virtual void notify_start_task(libutil::task_i *t) { }
    virtual void notify_finish_task(libutil::task_i *t);

};


} // unnamed namespace


template<size_t N, typename Traits>
void gen_bto_unfold_symmetry<N, Traits>::perform(
    gen_block_tensor_i<N, bti_traits> &bt) {

    try {

        symmetry<N, element_type> symcopy(bt.get_bis());

        {
            gen_block_tensor_ctrl<N, bti_traits> ctrl(bt);
            so_copy<N, element_type>(ctrl.req_const_symmetry()).
                perform(symcopy);
            ctrl.req_symmetry().clear();
        }

        gen_bto_unfold_symmetry_task_iterator<N, Traits> ti(bt, symcopy);
        gen_bto_unfold_symmetry_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        throw;
    }
}


template<size_t N, typename Traits>
void gen_bto_unfold_symmetry<N, Traits>::perform(
    const symmetry<N, element_type> &sym,
    const std::vector<size_t> &blst,
    gen_block_tensor_i<N, bti_traits> &bt) {

    try {

        gen_bto_unfold_symmetry_task_iterator<N, Traits> ti(blst, bt, sym);
        gen_bto_unfold_symmetry_task_observer<N, Traits> to;
        libutil::thread_pool::submit(ti, to);

    } catch(...) {
        throw;
    }
}


namespace {


template<size_t N, typename Traits>
gen_bto_unfold_symmetry_task<N, Traits>::gen_bto_unfold_symmetry_task(
    gen_block_tensor_i<N, bti_traits> &bt,
    const dimensions<N> &bidims,
    const symmetry<N, element_type> &symcopy,
    size_t aidx,
    bool orbits) :

    m_bt(bt), m_bidims(bidims), m_symcopy(symcopy), m_aidx(aidx),
    m_orbits(orbits) {

}


template<size_t N, typename Traits>
void gen_bto_unfold_symmetry_task<N, Traits>::perform() {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename bti_traits::template wr_block_type<N>::type wr_block_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy;

    gen_block_tensor_ctrl<N, bti_traits> ctrl(m_bt);

    if(m_orbits) {

        orbit<N, element_type> o(m_symcopy, m_aidx, false);

        rd_block_type &ba = ctrl.req_const_block(o.get_cindex());

        for(typename orbit<N, element_type>::iterator i = o.begin();
            i != o.end(); ++i) {

            if(o.get_abs_index(i) == m_aidx) continue;

            index<N> cidx;
            abs_index<N>::get_index(o.get_abs_index(i), m_bidims, cidx);
            wr_block_type &bb = ctrl.req_block(cidx);
            to_copy(ba, o.get_transf(i)).perform(true, bb);
            ctrl.ret_block(cidx);
        }

        ctrl.ret_const_block(o.get_cindex());

    } else {

        orbit<N, element_type> o(m_symcopy, m_aidx, false);

        if(o.get_acindex() != m_aidx) {

            index<N> idx;
            abs_index<N>::get_index(m_aidx, m_bidims, idx);

            if(ctrl.req_is_zero_block(idx) &&
                !ctrl.req_is_zero_block(o.get_cindex())) {

                rd_block_type &ba = ctrl.req_const_block(o.get_cindex());
                wr_block_type &bb = ctrl.req_block(idx);
                to_copy(ba, o.get_transf(m_aidx)).perform(true, bb);
                ctrl.ret_block(idx);
                ctrl.ret_const_block(o.get_cindex());
            }

        }
    }
}


template<size_t N, typename Traits>
gen_bto_unfold_symmetry_task_iterator<N, Traits>::
gen_bto_unfold_symmetry_task_iterator(
    gen_block_tensor_i<N, bti_traits> &bt,
    const symmetry<N, element_type> &symcopy) :

    m_bt(bt), m_bidims(m_bt.get_bis().get_block_index_dims()), m_ctrl(m_bt),
    m_symcopy(symcopy), m_orbits(true) {

    m_ctrl.req_nonzero_blocks(m_nzorb);
    m_i = m_nzorb.begin();
}


template<size_t N, typename Traits>
gen_bto_unfold_symmetry_task_iterator<N, Traits>::
gen_bto_unfold_symmetry_task_iterator(
    const std::vector<size_t> &blst,
    gen_block_tensor_i<N, bti_traits> &bt,
    const symmetry<N, element_type> &symcopy) :

    m_bt(bt), m_bidims(m_bt.get_bis().get_block_index_dims()), m_ctrl(m_bt),
    m_symcopy(symcopy), m_nzorb(blst), m_orbits(false) {

    m_i = m_nzorb.begin();
}


template<size_t N, typename Traits>
bool gen_bto_unfold_symmetry_task_iterator<N, Traits>::has_more() const {

    return m_i != m_nzorb.end();
}


template<size_t N, typename Traits>
libutil::task_i *gen_bto_unfold_symmetry_task_iterator<N, Traits>::get_next() {

    gen_bto_unfold_symmetry_task<N, Traits> *t =
        new gen_bto_unfold_symmetry_task<N, Traits>(m_bt, m_bidims, m_symcopy,
            *m_i, m_orbits);
    ++m_i;
    return t;
}


template<size_t N, typename Traits>
void gen_bto_unfold_symmetry_task_observer<N, Traits>::notify_finish_task(
    libutil::task_i *t) {

    delete t;
}


} // unnamed namespace

} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_UNFOLD_SYMMETRY_IMPL_H
