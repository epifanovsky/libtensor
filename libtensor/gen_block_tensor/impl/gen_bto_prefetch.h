#ifndef LIBTENSOR_GEN_BTO_PREFETCH_H
#define LIBTENSOR_GEN_BTO_PREFETCH_H

#include <vector>
#include <libtensor/core/abs_index.h>
#include "../gen_block_tensor_i.h"
#include "../gen_block_tensor_ctrl.h"

namespace libtensor {


template<size_t N, typename Traits>
class gen_bto_prefetch {
public:
    typedef typename Traits::bti_traits bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bt;
    dimensions<N> m_bidims;

public:
    gen_bto_prefetch(gen_block_tensor_rd_i<N, bti_traits> &bt) :
        m_bt(bt), m_bidims(m_bt.get_bis().get_block_index_dims())
    { }

    void perform(const std::vector<size_t> &blst);

};


template<size_t N, typename Traits>
void gen_bto_prefetch<N, Traits>::perform(const std::vector<size_t> &blst) {

    typedef typename bti_traits::template rd_block_type<N>::type rd_block_type;
    typedef typename Traits::template to_copy_type<N>::type to_copy;

    gen_block_tensor_rd_ctrl<N, bti_traits> ctrl(m_bt);

    for(typename std::vector<size_t>::const_iterator i = blst.begin();
        i != blst.end(); ++i) {

        index<N> bidx;
        abs_index<N>::get_index(*i, m_bidims, bidx);

        rd_block_type &blk = ctrl.req_const_block(bidx);
        to_copy(blk).prefetch();
        ctrl.ret_const_block(bidx);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_GEN_BTO_PREFETCH_H

