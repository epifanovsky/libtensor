#ifndef LIBTENSOR_BTOD_CONTRACT3_IMPL_H
#define LIBTENSOR_BTOD_CONTRACT3_IMPL_H

#include <libtensor/core/allocator.h>
#include <libtensor/core/block_tensor.h>
#include <libtensor/core/block_tensor_ctrl.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/block_tensor/btod/btod_contract2.h>
#include <libtensor/btod/btod_set.h>
#include "../btod_contract3.h"

namespace libtensor {


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
const char *btod_contract3<N1, N2, N3, K1, K2>::k_clazz =
    "btod_contract3<N1, N2, N3, K1, K2>";


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
btod_contract3<N1, N2, N3, K1, K2>::btod_contract3(
    const contraction2<N1, N2 + K2, K1> &contr1,
    const contraction2<N1 + N2, N3, K2> &contr2,
    block_tensor_i<N1 + K1, double> &bta,
    block_tensor_i<N2 + K1 + K2, double> &btb,
    block_tensor_i<N3 + K2, double> &btc) :

    m_contr1(contr1), m_contr2(contr2), m_bta(bta), m_btb(btb), m_btc(btc) {

}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
void btod_contract3<N1, N2, N3, K1, K2>::perform(
    block_tensor_i<N1 + N2 + N3, double> &btd) {

    block_tensor_ctrl<N1 + K1, double> ca(m_bta);
    block_tensor_ctrl<N2 + K1 + K2, double> cb(m_btb);

    //  Operation and buffer for the intermediate (AB)

    btod_contract2<N1, N2 + K2, K1> contrab(m_contr1, m_bta, m_btb);
    block_index_space<N1 + N2 + K2> bisab(contrab.get_bis());
    block_tensor< N1 + N2 + K2, double, allocator<double> > btab(bisab);
    block_tensor_ctrl<N1 + N2 + K2, double> cab(btab);
    {
        so_copy<N1 + N2 + K2, double>(contrab.get_symmetry()).
            perform(cab.req_symmetry());
    }

    //  Form batches of AB

    size_t batch_size = 64; // FIXME: arbitrary and hard-coded!
    bool first_batch = true;
    const assignment_schedule<N1 + N2 + K2, double> &schab =
        contrab.get_schedule();
    typename assignment_schedule<N1 + N2 + K2, double>::iterator isch =
        schab.begin();
    while(isch != schab.end()) {

        std::vector<size_t> batch;
        for(size_t i = 0; i < batch_size && isch != schab.end(); i++, ++isch) {
            batch.push_back(schab.get_abs_index(isch));
        }

        ca.req_sync_on();
        cb.req_sync_on();
        cab.req_sync_on();
        compute_batch_ab(contrab, batch, btab);
        ca.req_sync_off();
        cb.req_sync_off();
        cab.req_sync_off();

        if(first_batch) {
            btod_contract2<N1 + N2, N3, K2>(m_contr2, btab, m_btc).
                perform(btd);
            first_batch = false;
        } else {
            btod_contract2<N1 + N2, N3, K2>(m_contr2, btab, m_btc).
                perform(btd, 1.0);
        }
        btod_set<N1 + N2 + K2>().perform(btab);
    }
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
void btod_contract3<N1, N2, N3, K1, K2>::compute_batch_ab(
    btod_contract2<N1, N2 + K2, K1> &contr,
    const std::vector<size_t> &blst,
    block_tensor_i<N1 + N2 + K2, double> &btab) {

    block_tensor_ctrl<N1 + N2 + K2, double> cab(btab);
    dimensions<N1 + N2 + K2> bidims(btab.get_bis().get_block_index_dims());
    tensor_transf<N1 + N2 + K2, double> tr0;

    std::vector<batch_ab_task*> tl;

    for(typename std::vector<size_t>::const_iterator i = blst.begin();
        i != blst.end(); ++i) {

        abs_index<N1 + N2 + K2> aidx(*i, bidims);
        tl.push_back(new batch_ab_task(contr, aidx.get_index(), btab));
    }

    batch_ab_task_iterator ti(tl);
    batch_ab_task_observer to;
    libutil::thread_pool::submit(ti, to);

    for(size_t i = 0; i < tl.size(); i++) delete tl[i];
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
void btod_contract3<N1, N2, N3, K1, K2>::batch_ab_task::perform() {

    block_tensor_ctrl<N1 + N2 + K2, double> cab(m_btab);
    tensor_transf<N1 + N2 + K2, double> tr0;
    dense_tensor_i<N1 + N2 + K2, double> &blkab = cab.req_block(m_idx);
    m_contr.compute_block(true, blkab, m_idx, tr0, 1.0);
    cab.ret_block(m_idx);
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
bool
btod_contract3<N1, N2, N3, K1, K2>::batch_ab_task_iterator::has_more() const {

    return m_i != m_tl.end();
}


template<size_t N1, size_t N2, size_t N3, size_t K1, size_t K2>
libutil::task_i*
btod_contract3<N1, N2, N3, K1, K2>::batch_ab_task_iterator::get_next() {

    batch_ab_task *t = *m_i;
    ++m_i;
    return t;
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_CONTRACT3_IMPL_H

