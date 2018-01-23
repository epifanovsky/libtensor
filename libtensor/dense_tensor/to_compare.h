#ifndef LIBTENSOR_TO_COMPARE_H
#define LIBTENSOR_TO_COMPARE_H

#include "dense_tensor_i.h"
#include <libtensor/core/noncopyable.h>

namespace libtensor {

/** \brief Compares two tensors

    This operation takes two tensors with the same dimensions and
    compares each element. If the difference between the elements
    exceeds a given threshold, the tensors are considered different.

    The class can also return the information about the first found
    difference: the index and the two values.

    Example:
    \code
    dense_tensor_i<2,T> &t1(...), &t2(...);
    to_compare<2> comp(t1, t2, 1e-16);
    if(!comp.compare()) {
        index<2> idx(comp.get_diff_index());
        T elem1 = comp.get_diff_elem_1();
        T elem2 = comp.get_diff_elem_2();
        cout << "Difference found: "
            << "[" << idx[0] << "," << idx[1] << "]"
            << " " << elem1 << " vs " << elem2 << "." << endl;
    } else {
        cout << "No differences found." << endl;
    }
    \endcode

    \ingroup libtensor_tod
**/
template<size_t N, typename T>
class to_compare : public noncopyable {
public:
    static const char *k_clazz; //!< Class name

private:
    dense_tensor_rd_i<N, T> &m_t1; //!< First tensor
    dense_tensor_rd_i<N, T> &m_t2; //!< Second tensor
    T m_thresh; //!< Equality threshold
    index<N> m_idx_diff; //!< Index of the first different element
    T m_diff_elem_1; //!< Value of the first different element in t1
    T m_diff_elem_2; //!< Value of the first different element in t2

public:
    /** \brief Initializes the operation
        \param t1 First tensor.
        \param t2 Second tensor.
        \param thresh Threshold.

        The two tensors must have the same dimensions, otherwise an
        exception will be thrown.
    **/
    to_compare(dense_tensor_rd_i<N, T> &t1,
        dense_tensor_rd_i<N, T> &t2, T thresh);

    /** \brief Performs the comparison
        \return \c true if all the elements equal within the threshold,
            \c false otherwise
        \throw bad_dimensions if the two tensors have different dimensions.
    **/
    bool compare();

    /** \brief Returns the index of the first non-equal element
     **/
    const index<N> &get_diff_index() const {
        return m_idx_diff;
    }

    /** \brief Returns the value of the first different element in
            the first tensor
     **/
    T get_diff_elem_1() const {
        return m_diff_elem_1;
    }

    /** \brief Returns the value of the first different element in
            the second tensor
     **/
    T get_diff_elem_2() const {
        return m_diff_elem_2;
    }

};

template<size_t N>
using tod_compare = to_compare<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_TO_COMPARE_H
