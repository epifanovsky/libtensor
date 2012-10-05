#ifndef LIBTENSOR_DIAG_BTOD_RANDOM_IMPL_H
#define LIBTENSOR_DIAG_BTOD_RANDOM_IMPL_H

#include <libtensor/core/orbit_list.h>
#include <libtensor/diag_tensor/diag_tod_random.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "../diag_btod_random.h"

namespace libtensor {


template<size_t N>
class diag_btod_random_block {
public:
    typedef diag_block_tensor_i_traits<double> bti_traits;

private:
    gen_block_tensor_wr_ctrl<N, bti_traits> &m_ctrl;

public:
    diag_btod_random_block(gen_block_tensor_wr_ctrl<N, bti_traits> &ctrl) :
        m_ctrl(ctrl)
    { }

    void perform(const index<N> &bidx);

};


template<size_t N>
const char *diag_btod_random<N>::k_clazz = "diag_btod_random<N>";


template<size_t N>
void diag_btod_random<N>::perform(diag_block_tensor_wr_i<N, double> &bt) {

    typedef diag_block_tensor_i_traits<double> bti_traits;

    diag_btod_random::start_timer();

    try {

        gen_block_tensor_wr_ctrl<N, bti_traits> ctrl(bt);
        orbit_list<N, double> ol(ctrl.req_symmetry());
        for(typename orbit_list<N, double>::iterator io = ol.begin();
            io != ol.end(); ++io) {
            diag_btod_random_block<N>(ctrl).perform(ol.get_index(io));
        }

    } catch(...) {
        diag_btod_random::stop_timer();
        throw;
    }

    diag_btod_random::stop_timer();
}

template<size_t N>
void diag_btod_random<N>::perform(
    diag_block_tensor_wr_i<N, double> &bt,
    const index<N> &bidx) {

    typedef diag_block_tensor_i_traits<double> bti_traits;

    diag_btod_random::start_timer();

    try {

        gen_block_tensor_wr_ctrl<N, bti_traits> ctrl(bt);
        diag_btod_random_block<N>(ctrl).perform(bidx);

    } catch(...) {
        diag_btod_random::stop_timer();
        throw;
    }

    diag_btod_random::stop_timer();
}


template<size_t N>
void diag_btod_random_block<N>::perform(const index<N> &bidx) {

    diag_tensor_wr_i<N, double> &blk = m_ctrl.req_block(bidx);
    diag_tod_random<N>().perform(blk);
    m_ctrl.ret_block(bidx);
}


} // namespace libtensor

#endif // LIBTENSOR_DIAG_BTOD_RANDOM_IMPL_H
