#ifndef LIBTENSOR_TO_SELECT_IMPL_H
#define LIBTENSOR_TO_SELECT_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../to_select.h"

namespace libtensor {


template<size_t N, typename T, typename ComparePolicy>
void to_select<N, T, ComparePolicy>::perform(list_type &li, size_t n) {

    if (n == 0) return;

    dense_tensor_rd_ctrl<N, T> ctrl(m_t);
    const dimensions<N> &d = m_t.get_dims();
    const T *p = ctrl.req_const_dataptr();

    bool do_perm = !m_perm.is_identity();

    size_t i = 0;
    while (i < d.get_size() && p[i] == 0.0) i++;

    if (i == d.get_size()) {
        ctrl.ret_const_dataptr(p);
        return;
    }

    if (li.empty()) {
        abs_index<N> aidx(i, d);
        index<N> idx(aidx.get_index());
        if (do_perm) idx.permute(m_perm);
        li.insert(li.end(), tensor_element_type(idx, m_c * p[i]));
        i++;
    }

    for (; i < d.get_size(); i++) {
        //ignore zero elements
        if (p[i] == 0.0) continue;

        T val = p[i] * m_c;

        if (! m_cmp(val, li.back().get_value())) {
            if (li.size() < n) {
                abs_index<N> aidx(i, d);
                index<N> idx(aidx.get_index());
                if (do_perm) idx.permute(m_perm);
                li.push_back(tensor_element_type(idx, val));
            }
        }
        else {
            if (li.size() == n) li.pop_back();

            typename list_type::iterator it = li.begin();
            while (it != li.end() && ! m_cmp(val, it->get_value())) it++;
            abs_index<N> aidx(i, d);
            index<N> idx(aidx.get_index());
            if (do_perm) idx.permute(m_perm);
            li.insert(it, tensor_element_type(idx, val));
        }
    }

    ctrl.ret_const_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TO_SELECT_IMPL_H
