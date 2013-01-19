#include <libtensor/tod/kernels/cuda_kern_copy_generic.h>
#include "../cuda_tod_copy.h"

namespace libtensor {


template<size_t N>
const char *cuda_tod_copy<N>::k_clazz = "cuda_tod_copy<N>";


template<size_t N>
cuda_tod_copy<N>::cuda_tod_copy(dense_tensor_i<N, double> &ta, double c) :

    m_ta(ta), m_c(c), m_dimsb(mk_dimsb(m_ta, m_perm)) {

}


template<size_t N>
cuda_tod_copy<N>::cuda_tod_copy(dense_tensor_i<N, double> &ta,
    const permutation<N> &p, double c) :

    m_ta(ta), m_perm(p), m_c(c), m_dimsb(mk_dimsb(ta, p)) {

}


template<size_t N>
void cuda_tod_copy<N>::prefetch() {

    dense_tensor_ctrl<N, double>(m_ta).req_prefetch();
}


template<size_t N>
void cuda_tod_copy<N>::perform(dense_tensor_i<N, double> &tb) {

    static const char *method = "perform(dense_tensor_i<N, double>&)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }

    do_perform(tb, 0);
}


template<size_t N>
void cuda_tod_copy<N>::perform(dense_tensor_i<N, double> &tb, double c) {

    static const char *method = "perform(dense_tensor_i<N, double>&, double)";

    if(!tb.get_dims().equals(m_dimsb)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__, "tb");
    }
    if(c == 0) return;

    do_perform(tb, c);
}


template<size_t N>
dimensions<N> cuda_tod_copy<N>::mk_dimsb(dense_tensor_i<N, double> &ta,
    const permutation<N> &perm) {

    dimensions<N> dims(ta.get_dims());
    dims.permute(perm);
    return dims;
}


template<size_t N>
void cuda_tod_copy<N>::do_perform(dense_tensor_i<N, double> &tb, double c) {

    cuda_tod_copy::start_timer();

    try {

    dense_tensor_ctrl<N, double> ca(m_ta), cb(tb);
    ca.req_prefetch();
    cb.req_prefetch();

    const dimensions<N> &dimsa = m_ta.get_dims();

    const double *pa = ca.req_const_dataptr();
    double *pb = cb.req_dataptr();

    cuda_kern_copy_generic *kern = cuda_kern_copy_generic::match(
        pa, pb, dimsa, m_perm, m_c, c);
    cuda_tod_copy::start_timer(kern->get_name());
    kern->run();
    cuda_tod_copy::stop_timer(kern->get_name());
    delete kern; kern = 0;

    ca.ret_const_dataptr(pa);
    cb.ret_dataptr(pb);

    } catch(...) {
        cuda_tod_copy::stop_timer();
        throw;
    }

    cuda_tod_copy::stop_timer();
}


} // namespace libtensor

