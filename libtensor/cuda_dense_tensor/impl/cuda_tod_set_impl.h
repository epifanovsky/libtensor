#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include "../cuda_tod_set.h"
#include "cuda_kern_set.h"

namespace libtensor {


template<size_t N>
cuda_tod_set<N>::cuda_tod_set(double v) : m_v(v) {

}


template<size_t N>
void cuda_tod_set<N>::perform(dense_tensor_wr_i<N, double> &t) {

    dense_tensor_wr_ctrl<N, double> ctrl(t);
    double *d = ctrl.req_dataptr();
    double *p = d;

     size_t sz = t.get_dims().get_size();
     size_t grid, threads;
//     threads = 65535;
     threads = 1024;
//     size_t max_threads_per_kernel = 1024 * threads;
     size_t max_threads_per_kernel = 65535 * threads;
     for(size_t i = 0; i < sz; i+= max_threads_per_kernel) {
         size_t remaining = 0;
        //Do max possible number of blocks if possible
         if (i + max_threads_per_kernel <= sz) {
             grid = max_threads_per_kernel/threads;
             cuda::kern_set<<<grid,threads>>>(p, m_v);
         //if not do remaining number
         } else {
             grid = (sz - i)/threads;
             cuda::kern_set<<<grid,threads>>>(p, m_v);
             remaining = (sz - i)%threads;
             if (remaining) {
                 cuda::kern_set<<<1, remaining>>>(p + grid*threads, m_v);
             }
         }
         p += max_threads_per_kernel;
     }

    ctrl.ret_dataptr(d);
}


} // namespace libtensor

