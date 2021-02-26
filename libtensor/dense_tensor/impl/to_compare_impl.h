#ifndef LIBTENSOR_TO_COMPARE_IMPL_H
#define LIBTENSOR_TO_COMPARE_IMPL_H

#include <cmath> // for fabs // FIXME std::abs may be better, since it supports several types
#include <libtensor/core/abs_index.h>
#include <libtensor/core/bad_dimensions.h>
#include "../dense_tensor_ctrl.h"
#include "../to_compare.h"

namespace libtensor {


template<size_t N, typename T>
const char *to_compare<N, T>::k_clazz = "to_compare<N, T>";


template<size_t N, typename T>
to_compare<N, T>::to_compare(dense_tensor_rd_i<N, T> &t1,
    dense_tensor_rd_i<N, T> &t2, T thresh) :

    m_t1(t1), m_t2(t2), m_thresh(fabs(thresh)),
    m_diff_elem_1(0.0), m_diff_elem_2(0.0) {

    static const char *method = "to_compare(dense_tensor_rd_i<N, T>&, "
        "dense_tensor_rd_i<N, T>&, T)";

    const dimensions<N> &dims1(m_t1.get_dims()), &dims2(m_t2.get_dims());
    if(!dims1.equals(dims2)) {
        throw bad_dimensions(g_ns, k_clazz, method, __FILE__, __LINE__,
            "dims(t1) != dims(t2)");
    }
}


template<size_t N, typename T>
bool to_compare<N, T>::compare() {

    dense_tensor_rd_ctrl<N, T> tctrl1(m_t1), tctrl2(m_t2);
    const T *p1 = tctrl1.req_const_dataptr();
    const T *p2 = tctrl2.req_const_dataptr();

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

#endif // LIBTENSOR_TO_COMPARE_IMPL_H
