#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_dimensions.h>
#include <libtensor/linalg/linalg.h>
#include <libtensor/kernels/kern_dcopy.h>
#include <libtensor/kernels/loop_list_runner.h>
#include "../dense_tensor_ctrl.h"
#include "../to_copy_wnd.h"

namespace libtensor {


template<size_t N, typename T>
const char to_copy_wnd<N, T>::k_clazz[] = "to_copy_wnd<N, T>";


template<size_t N, typename T>
to_copy_wnd<N, T>::to_copy_wnd(
    dense_tensor_rd_i<N, T> &ta, const index_range<N> &ira) :

    m_ta(ta), m_ira(ira) {

}


template<size_t N, typename T>
void to_copy_wnd<N, T>::perform(
    dense_tensor_wr_i<N, T> &tb, const index_range<N> &irb) {

    static const char method[] = "perform(dense_tensor_wr_i<N, T>&, "
        "const index_range<N>&)";

    dimensions<N> dimsa(m_ira), dimsb(irb);
    if(!dimsa.equals(dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
    //  Check whether ta.dims contains m_ira, tb.dims contains irb

    const dimensions<N> &dimsa0 = m_ta.get_dims();
    const dimensions<N> &dimsb0 = tb.get_dims();

    try {

        dense_tensor_rd_ctrl<N, T> ca(m_ta);
        dense_tensor_wr_ctrl<N, T> cb(tb);

        ca.req_prefetch();
        cb.req_prefetch();

        std::list< loop_list_node<1, 1> > lpcopy1, lpcopy2;
        typename std::list< loop_list_node<1, 1> >::iterator icopy =
            lpcopy1.end();

        for(size_t i = 0; i < N; i++) {
            icopy = lpcopy1.insert(lpcopy1.end(),
                loop_list_node<1, 1>(dimsa[i]));
            icopy->stepa(0) = dimsa0.get_increment(i);
            icopy->stepb(0) = dimsb0.get_increment(i);
        }

        const T *pa = ca.req_const_dataptr();
        T *pb = cb.req_dataptr();

        size_t offa = abs_index<N>::get_abs_index(m_ira.get_begin(), dimsa0);
        size_t offb = abs_index<N>::get_abs_index(irb.get_begin(), dimsb0);
        size_t offenda = abs_index<N>::get_abs_index(m_ira.get_end(), dimsa0);
        size_t offendb = abs_index<N>::get_abs_index(irb.get_end(), dimsb0);
        loop_registers_x<1, 1, T> r;
        r.m_ptra[0] = pa + offa;
        r.m_ptrb[0] = pb + offb;
        r.m_ptra_end[0] = pa + offenda + 1;
        r.m_ptrb_end[0] = pb + offendb + 1;

        {
            std::unique_ptr< kernel_base<linalg, 1, 1, T> > kern(
                kern_copy<linalg, T>::match(1.0, lpcopy1, lpcopy2));
            loop_list_runner_x<linalg, 1, 1, T>(lpcopy1).run(0, r, *kern);
        }

        ca.ret_const_dataptr(pa);
        cb.ret_dataptr(pb);

    } catch(...) {
        throw;
    }
}


} // namespace libtensor

