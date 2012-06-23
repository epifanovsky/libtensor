#ifndef LIBTENSOR_SO_APPLY_IMPL_H
#define LIBTENSOR_SO_APPLY_IMPL_H

namespace libtensor {

template<size_t N, typename T>
void so_apply<N, T>::perform(symmetry<N, T> &sym2) {

    sym2.clear();

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);
        symmetry_element_set<N, T> set2(set1.get_id());

        symmetry_operation_params<operation_t> params(
                set1, m_perm1, m_s1, m_s2, m_keep_zero, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_H

