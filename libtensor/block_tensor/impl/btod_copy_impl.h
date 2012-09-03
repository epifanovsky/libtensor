#ifndef LIBTENSOR_BTOD_COPY_IMPL_H
#define LIBTENSOR_BTOD_COPY_IMPL_H

#include <libtensor/dense_tensor/tod_copy.h>
#include <libtensor/dense_tensor/tod_set.h>
#include <libtensor/gen_block_tensor/impl/gen_bto_copy_impl.h>
#include <libtensor/block_tensor/block_tensor_ctrl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_add_impl.h>
#include <libtensor/block_tensor/bto/impl/bto_aux_copy_impl.h>
#include "../btod_copy.h"

namespace libtensor {


template<size_t N>
const char *btod_copy<N>::k_clazz = "btod_copy<N>";


template<size_t N, typename Traits>
class bto_stream_adapter : public gen_block_stream_i<N, typename Traits::bti_traits> {
public:
    typedef typename Traits::bti_traits bti_traits;
    typedef typename bti_traits::element_type element_type;
    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;

private:
    bto_stream_i<N, Traits> &m_out;

public:
    bto_stream_adapter(bto_stream_i<N, Traits> &out) : m_out(out) { }
    virtual ~bto_stream_adapter() { }
    virtual void open() { m_out.open(); }
    virtual void close() { m_out.close(); }
    virtual void put(
        const index<N> &idx,
        rd_block_type &blk,
        const tensor_transf<N, element_type> &tr) {
        m_out.put(idx, blk, tr);
    }

};


template<size_t N>
void btod_copy<N>::perform(bto_stream_i<N, btod_traits> &out) {

    bto_stream_adapter<N, btod_traits> a(out);
    m_gbto.perform(a);
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb) {

    bto_aux_copy<N, btod_traits> out(get_symmetry(), btb);
    perform(out);
}


template<size_t N>
void btod_copy<N>::perform(block_tensor_i<N, double> &btb, const double &c) {

    block_tensor_ctrl<N, double> cb(btb);
    addition_schedule<N, btod_traits> asch(get_symmetry(),
        cb.req_const_symmetry());
    asch.build(get_schedule(), cb);

    bto_aux_add<N, btod_traits> out(get_symmetry(), asch, btb, c);
    perform(out);
}


template<size_t N>
void btod_copy<N>::compute_block(bool zero, dense_tensor_i<N, double> &blkb,
    const index<N> &ib, const tensor_transf<N, double> &trb, const double &c) {

    m_gbto.compute_block(zero, blkb, ib, trb, c);
}


} // namespace libtensor

#endif // LIBTENSOR_BTOD_COPY_IMPL_H
