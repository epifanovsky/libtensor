#ifndef LIBTENSOR_SO_SYMMETRIZE_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_IMPL_H

namespace libtensor {

template<size_t N, typename T>
void so_symmetrize<N, T>::perform(symmetry<N, T> &sym2) {

    sym2.clear();

    bool perm_done = false;
    for(typename symmetry<N, T>::iterator i1 = m_sym1.begin();
            i1 != m_sym1.end(); i1++) {

        const symmetry_element_set<N, T> &set1 =
                m_sym1.get_subset(i1);
        if(set1.get_id().compare(se_perm<N, T>::k_sym_type) == 0) {
            perm_done = true;
        }

        symmetry_element_set<N, T> set2(set1.get_id());
        symmetry_operation_params<operation_t> params(
                set1, m_perm, m_symm, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }

    //	If the argument does not have any permutation symmetry
    //	elements, the handler does not get called above. We need to
    //	fix that manually.
    if(!perm_done) {
        symmetry_element_set<N, T> set1(se_perm<N, T>::k_sym_type),
                set2(se_perm<N, T>::k_sym_type);
        symmetry_operation_params<operation_t> params(
                set1, m_perm, m_symm, set2);
        dispatcher_t::get_instance().invoke(set1.get_id(), params);

        for(typename symmetry_element_set<N, T>::iterator j =
                set2.begin(); j != set2.end(); j++) {
            sym2.insert(set2.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_IMPL_H
