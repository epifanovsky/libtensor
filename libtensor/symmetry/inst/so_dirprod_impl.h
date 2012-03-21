#ifndef LIBTENSOR_SO_DIRPROD_IMPL_H
#define LIBTENSOR_SO_DIRPROD_IMPL_H

namespace libtensor {

template<size_t N, size_t M, typename T>
void so_dirprod<N, M, T>::perform(symmetry<N + M, T> &sym3) {

    sym3.clear();

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

        typename symmetry<M, T>::iterator j;
        for(j = m_sym2.begin(); j != m_sym2.end(); j++) {
            if(set1.get_id() == m_sym2.get_subset(j).get_id()) break;
        }

        symmetry_element_set<N + M, T> set3(set1.get_id());

        if(j == m_sym2.end()) {
            symmetry_element_set<M, T> set2(set1.get_id());
            symmetry_operation_params<operation_t> params(set1, set2,
                    m_perm, sym3.get_bis(), set3);
            dispatcher_t::get_instance().invoke(set1.get_id(), params);
        } else {
            const symmetry_element_set<M, T> &set2 = m_sym2.get_subset(j);
            symmetry_operation_params<operation_t> params(set1, set2,
                    m_perm, sym3.get_bis(), set3);
            dispatcher_t::get_instance().invoke(set1.get_id(), params);
        }

        copy_subset(set3, sym3);
    }

    for(typename symmetry<M, T>::iterator i = m_sym2.begin();
            i != m_sym2.end(); i++) {

        const symmetry_element_set<M, T> &set2 = m_sym2.get_subset(i);

        typename symmetry<N, T>::iterator j;
        for(j = m_sym1.begin(); j != m_sym1.end(); j++) {

            if(set2.get_id() == m_sym1.get_subset(j).get_id()) break;
        }

        if (j != m_sym1.end()) continue;

        symmetry_element_set<N + M, T> set3(set2.get_id());

        symmetry_element_set<N, T> set1(set2.get_id());
        symmetry_operation_params<operation_t> params(set1, set2,
                m_perm, sym3.get_bis(), set3);
        dispatcher_t::get_instance().invoke(set2.get_id(), params);

        copy_subset(set3, sym3);
    }
}


template<size_t N, size_t M, typename T>
template<size_t X>
void so_dirprod<N, M, T>::copy_subset(const symmetry_element_set<X, T> &set1,
        symmetry<X, T> &sym2) {

    for(typename symmetry_element_set<X, T>::const_iterator i =
            set1.begin(); i != set1.end(); i++) {

        sym2.insert(set1.get_elem(i));
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_IMPL_H
