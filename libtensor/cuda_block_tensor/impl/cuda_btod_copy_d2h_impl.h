#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_d2h.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "../cuda_btod_copy_d2h.h"

namespace libtensor {


template<size_t N>
const char cuda_btod_copy_d2h<N>::k_clazz[] = "cuda_btod_copy_d2h<N>";


template<size_t N>
void cuda_btod_copy_d2h<N>::perform(block_tensor_wr_i<N, double> &bth) {

    static const char method[] = "perform(cuda_block_tensor_wr_i<N, double>&)";

    if(!m_btd.get_bis().equals(bth.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "bth");
    }

    typedef block_tensor_i_traits<double> bti_traits;
    typedef cuda_block_tensor_i_traits<double> cbti_traits;

    cuda_btod_copy_d2h::start_timer();

    try {

        gen_block_tensor_rd_ctrl<N, cbti_traits> ca(m_btd);
        dimensions<N> bidimsa = m_btd.get_bis().get_block_index_dims();

        std::vector<size_t> nzorba;
        ca.req_nonzero_blocks(nzorba);

        gen_block_tensor_wr_ctrl<N, bti_traits> cb(bth);

        cb.req_zero_all_blocks();
        so_copy<N, double>(ca.req_const_symmetry()).perform(cb.req_symmetry());

        for(size_t i = 0; i < nzorba.size(); i++) {
            index<N> bi;
            abs_index<N>::get_index(nzorba[i], bidimsa, bi);
            dense_tensor_rd_i<N, double> &blka = ca.req_const_block(bi);
            dense_tensor_wr_i<N, double> &blkb = cb.req_block(bi);
            cuda_tod_copy_d2h<N>(blka).perform(blkb);
        }

    } catch(...) {
        cuda_btod_copy_d2h::stop_timer();
        throw;
    }

    cuda_btod_copy_d2h::stop_timer();
}


} // namespace libtensor

