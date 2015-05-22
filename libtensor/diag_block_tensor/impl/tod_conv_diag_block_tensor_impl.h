#ifndef LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_IMPL_H
#define LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_IMPL_H

#include <libtensor/core/allocator.h>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/orbit.h>
#include <libtensor/core/orbit_list.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/diag_tensor/diag_tensor.h>
#include <libtensor/diag_tensor/diag_tod_copy.h>
#include <libtensor/diag_tensor/tod_conv_diag_tensor.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include <libtensor/diag_block_tensor/diag_btod_traits.h>
#include "../diag_btod_traits.h"
#include "../tod_conv_diag_block_tensor.h"

namespace libtensor {


template<size_t N>
void tod_conv_diag_block_tensor<N>::perform(dense_tensor_wr_i<N, double> &tb) {

    typedef allocator<double> allocator_t;
    typedef diag_block_tensor_i_traits<double> bti_traits;
    typedef typename diag_btod_traits::template temp_block_tensor_type<N>::type
        temp_block_tensor_type;

    gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bta);
    const block_index_space<N> &bisa = m_bta.get_bis();
    dimensions<N> bidimsa = bisa.get_block_index_dims();
    const symmetry<N, double> &syma = ca.req_const_symmetry();

    tod_set<N>().perform(true, tb);

    orbit_list<N, double> ol(syma);
    for(typename orbit_list<N, double>::iterator io = ol.begin();
        io != ol.end(); ++io) {

        temp_block_tensor_type btb(bisa);
        gen_block_tensor_ctrl<N, bti_traits> cb(btb);

        index<N> ia;
        ol.get_index(io, ia);
        orbit<N, double> oa(syma, ia);
        diag_tensor_rd_i<N, double> &blka = ca.req_const_block(ia);
        for(typename orbit<N, double>::iterator ioa = oa.begin();
            ioa != oa.end(); ++ioa) {

            index<N> ib;
            abs_index<N>::get_index(oa.get_abs_index(ioa), bidimsa, ib);
            index<N> off = bisa.get_block_start(ib);

            {
                diag_tensor_wr_i<N, double> &blkb = cb.req_block(ib);
                diag_tod_copy<N>(blka, oa.get_transf(ioa)).perform(true, blkb);
                cb.ret_block(ib);
            }
            {
                diag_tensor_rd_i<N, double> &blkb = cb.req_const_block(ib);
                tod_conv_diag_tensor<N>(blkb).perform(tb, off);
                cb.ret_const_block(ib);
            }

            cb.req_zero_block(ib);
        }
        ca.ret_const_block(ia);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_CONV_DIAG_BLOCK_TENSOR_IMPL_H
