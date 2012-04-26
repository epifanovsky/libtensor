#ifndef LIBTENSOR_SO_COPY_IMPL_H
#define LIBTENSOR_SO_COPY_IMPL_H

namespace libtensor {

template<size_t N, typename T>
void so_copy<N, T>::perform(symmetry<N, T> &sym2) {

    sym2.clear();

    for(typename symmetry<N, T>::iterator i = m_sym1.begin();
            i != m_sym1.end(); i++) {

        const symmetry_element_set<N, T> &set1 = m_sym1.get_subset(i);

        for(typename symmetry_element_set<N, T>::const_iterator j =
                set1.begin(); j != set1.end(); j++) {

            sym2.insert(set1.get_elem(j));
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_COPY_IMPL_H
