#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_block_index_space.h>
#include <libtensor/symmetry/so_copy.h>
#include <libtensor/cuda_dense_tensor/cuda_tod_copy_h2d.h>
#include <libtensor/gen_block_tensor/gen_block_tensor_ctrl.h>
#include "../cuda_btod_copy_h2d.h"

namespace libtensor {


template<size_t N>
const char cuda_btod_copy_h2d<N>::k_clazz[] = "cuda_btod_copy_h2d<N>";


template<size_t N>
void cuda_btod_copy_h2d<N>::perform(cuda_block_tensor_wr_i<N, double> &btd) {

    static const char method[] = "perform(cuda_block_tensor_wr_i<N, double>&)";

    if(!m_bth.get_bis().equals(btd.get_bis())) {
        throw bad_block_index_space(g_ns, k_clazz, method, __FILE__, __LINE__,
            "btd");
    }

    typedef block_tensor_i_traits<double> bti_traits;
    typedef cuda_block_tensor_i_traits<double> cbti_traits;

    cuda_btod_copy_h2d::start_timer();

    try {

        gen_block_tensor_rd_ctrl<N, bti_traits> ca(m_bth);
        dimensions<N> bidimsa = m_bth.get_bis().get_block_index_dims();

        std::vector<size_t> nzorba;
        ca.req_nonzero_blocks(nzorba);

        gen_block_tensor_wr_ctrl<N, cbti_traits> cb(btd);

        cb.req_zero_all_blocks();
        so_copy<N, double>(ca.req_const_symmetry()).perform(cb.req_symmetry());

        for(size_t i = 0; i < nzorba.size(); i++) {
            index<N> bi;
            abs_index<N>::get_index(nzorba[i], bidimsa, bi);
            dense_tensor_rd_i<N, double> &blka = ca.req_const_block(bi);
            dense_tensor_wr_i<N, double> &blkb = cb.req_block(bi);
            cuda_tod_copy_h2d<N>(blka).perform(blkb);
        }

    } catch(...) {
        cuda_btod_copy_h2d::stop_timer();
        throw;
    }

    cuda_btod_copy_h2d::stop_timer();
}


} // namespace libtensor

