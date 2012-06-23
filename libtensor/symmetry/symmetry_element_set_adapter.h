#ifndef LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_H
#define LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_H

#include <typeinfo>
#include "../core/symmetry_element_set.h"
#include "bad_symmetry.h"

namespace libtensor {


/** \brief Adapts symmetry_element_set<N, T> to a particular symmetry
        element type

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T, typename ElemT>
class symmetry_element_set_adapter {
public:
    static const char *k_clazz; //!< Class name;

public:
    //!    Iterator type
    typedef typename symmetry_element_set<N, T>::const_iterator iterator;

private:
    const symmetry_element_set<N, T> &m_set; //!< Element set

public:
    symmetry_element_set_adapter(const symmetry_element_set<N, T> &set) :
        m_set(set) { }

    bool is_empty() const {
        return m_set.is_empty();
    }

    iterator begin() const {
        return m_set.begin();
    }

    iterator end() const {
        return m_set.end();
    }

    const ElemT &get_elem(iterator &i) const;
};


template<size_t N, typename T, typename ElemT>
const char *symmetry_element_set_adapter<N, T, ElemT>::k_clazz =
        "symmetry_element_set_adapter<N, T, ElemT>";


template<size_t N, typename T, typename ElemT>
const ElemT &symmetry_element_set_adapter<N, T, ElemT>::get_elem(
        iterator &i) const {

    static const char *method = "get_elem(iterator&)";

    try {
        return dynamic_cast<const ElemT&>(m_set.get_elem(i));
    } catch(std::bad_cast&) {
        throw bad_symmetry(
                g_ns, k_clazz, method, __FILE__, __LINE__, "bad_cast");
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SYMMETRY_ELEMENT_SET_ADAPTER_H

