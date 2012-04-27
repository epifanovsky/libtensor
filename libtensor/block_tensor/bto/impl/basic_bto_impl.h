#ifndef LIBTENSOR_BASIC_BTO_IMPL_H
#define LIBTENSOR_BASIC_BTO_IMPL_H

#include <libtensor/symmetry/so_copy.h>

namespace libtensor {


template<size_t N, typename Traits>
void basic_bto<N, Traits>::perform(block_tensor_t &bt) {

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;

    sync_on();

    block_tensor_ctrl_t ctrl(bt);
    ctrl.req_sync_on();
    ctrl.req_zero_all_blocks();
    so_copy<N, element_t> (get_symmetry()).perform(ctrl.req_symmetry());

    std::vector<task*> tasks;

    dimensions<N> bidims(bt.get_bis().get_block_index_dims());
    const assignment_schedule<N, element_t> &sch = get_schedule();
    for(typename assignment_schedule<N, element_t>::iterator i = sch.begin();
        i != sch.end(); ++i) {

        task *t = new task(*this, bt, bidims, sch, i);
        tasks.push_back(t);
    }

    task_iterator ti(tasks);
    task_observer to;
    libutil::thread_pool::submit(ti, to);

    for(typename std::vector<task*>::iterator i = tasks.begin();
        i != tasks.end(); ++i) {
        delete *i;
    }
    tasks.clear();
    ctrl.req_sync_off();

    sync_off();
}


template<size_t N, typename Traits>
void basic_bto<N, Traits>::task::perform() {

    cpu_pool cpus(1);

    typedef typename Traits::template block_tensor_ctrl_type<N>::type
        block_tensor_ctrl_t;
    typedef typename Traits::template block_type<N>::type block_t;

    block_tensor_ctrl_t ctrl(m_bt);
    abs_index<N> ai(m_sch.get_abs_index(m_i), m_bidims);
    block_t &blk = ctrl.req_block(ai.get_index());
    m_bto.compute_block(blk, ai.get_index(), cpus);
    ctrl.ret_block(ai.get_index());
}


template<size_t N, typename Traits>
bool basic_bto<N, Traits>::task_iterator::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N, typename Traits>
libutil::task_i *basic_bto<N, Traits>::task_iterator::get_next() {

    libutil::task_i *t = *m_i;
    ++m_i;
    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_BASIC_BTO_IMPL_H
