#ifndef LIBTENSOR_TOD_COMPARE_IMPL_H
#define LIBTENSOR_TOD_COMPARE_IMPL_H

#include <cmath> // for fabs
#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../tod_compare.h"

namespace libtensor {


template<size_t N>
const char *tod_compare<N>::k_clazz = "tod_compare<N>";


template<size_t N>
tod_compare<N>::tod_compare(dense_tensor_rd_i<N, double> &t1,
    dense_tensor_rd_i<N, double> &t2, double thresh) :

    m_t1(t1), m_t2(t2), m_thresh(fabs(thresh)),
    m_diff_elem_1(0.0), m_diff_elem_2(0.0) {

    static const char *method = "tod_compare(dense_tensor_rd_i<N, double>&, "
        "dense_tensor_rd_i<N, double>&, double)";

    const dimensions<N> &dims1(m_t1.get_dims()), &dims2(m_t2.get_dims());
    if(!dims1.equals(dims2)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dims(t1) != dims(t2)");
    }
}


template<size_t N>
bool tod_compare<N>::compare() {

    dense_tensor_rd_ctrl<N, double> tctrl1(m_t1), tctrl2(m_t2);
    const double *p1 = tctrl1.req_const_dataptr();
    const double *p2 = tctrl2.req_const_dataptr();

    for(size_t i = 0; i < N; i++) m_idx_diff[i] = 0;
    size_t sz = m_t1.get_dims().get_size();
    bool equal = true;
    abs_index<N> idx(m_t1.get_dims());
    for(size_t i = 0; i < sz; i++) {
        if(fabs(p1[i]) <= 1.0) {
            if(fabs(p1[i] - p2[i]) > m_thresh) {
                m_diff_elem_1 = p1[i];
                m_diff_elem_2 = p2[i];
                equal = false;
                break;
            }
        } else {
            if(fabs(p2[i]/p1[i] - 1.0) > m_thresh) {
                m_diff_elem_1 = p1[i];
                m_diff_elem_2 = p2[i];
                equal = false;
                break;
            }
        }
        idx.inc();
    }
    if(!equal) m_idx_diff = idx.get_index();

    tctrl1.ret_const_dataptr(p1);
    tctrl2.ret_const_dataptr(p2);

    return equal;
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_COMPARE_IMPL_H
