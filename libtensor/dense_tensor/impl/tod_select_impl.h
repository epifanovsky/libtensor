#ifndef LIBTENSOR_TOD_SELECT_IMPL_H
#define LIBTENSOR_TOD_SELECT_IMPL_H

#include "../dense_tensor_ctrl.h"
#include "../tod_select.h"

namespace libtensor {


template<size_t N, typename ComparePolicy>
void tod_select<N, ComparePolicy>::perform(list_t &li, size_t n) {

    if (n == 0) return;

    dense_tensor_rd_ctrl<N, double> ctrl(m_t);
    const dimensions<N> &d = m_t.get_dims();
    const double *p = ctrl.req_const_dataptr();

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
        li.insert(li.end(), elem_t(idx, m_c * p[i]));
        i++;
    }

    for (; i < d.get_size(); i++) {
        //ignore zero elements
        if (p[i] == 0.0) continue;

        double val = p[i] * m_c;

        if (! m_cmp(val, li.back().value)) {
            if (li.size() < n) {
                abs_index<N> aidx(i, d);
                index<N> idx(aidx.get_index());
                if (do_perm) idx.permute(m_perm);
                li.push_back(elem_t(idx, val));
            }
        }
        else {
            if (li.size() == n) li.pop_back();

            typename list_t::iterator it = li.begin();
            while (it != li.end() && ! m_cmp(val, it->value)) it++;
            abs_index<N> aidx(i, d);
            index<N> idx(aidx.get_index());
            if (do_perm) idx.permute(m_perm);
            li.insert(it, elem_t(idx, val));
        }
    }

    ctrl.ret_const_dataptr(p);
}


} // namespace libtensor

#endif // LIBTENSOR_TOD_SELECT_IMPL_H
