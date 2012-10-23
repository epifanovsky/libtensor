#ifndef LIBTENSOR_TOD_SELECT_H
#define LIBTENSOR_TOD_SELECT_H

#include <cmath>
#include <list>
#include <libtensor/core/abs_index.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_element.h>
#include <libtensor/core/tensor_transf.h>

namespace libtensor {


//! \name Common compare policies
//@{

struct compare4max {
    bool operator()(double a, double b) {
        return a > b;
    }
};

struct compare4absmax {
    bool operator()(double a, double b) {
        return fabs(a) > fabs(b);
    }
};

struct compare4min {
    bool operator()(double a, double b) {
        return a < b;
    }
};

struct compare4absmin {
    bool operator()(double a, double b) {
        return fabs(a) < fabs(b);
    }
};

//@}


/** \brief Selects a number of elements from a tensor
    \tparam N Tensor order.
    \tparam ComparePolicy Policy to select elements.

    The operation selects a number of elements from the tensor and adds them
    as (index, value) to a given list. The elements are selected by the
    ordering imposed on the elements by the compare policy. Zero elements are
    never selected. The resulting list of elements is ordered according to the
    compare policy.

    If a permutation and / or a coefficient are given in the construct, the
    tensor elements are permuted and scaled before the list is constructed
    (this does not affect the input tensor).

    <b>Compare policy</b>

    The compare policy type determines the ordering of tensor elements by which
    they are selected. Any type used as compare policy needs to implement a
    function
    <code>
    bool operator()(double a, double b)
    </code>
    which compares two tensor elements. If the function returns true, the first
    value is taken as the more optimal with respect to the compare policy.

    \sa compare4max, compare4min, compare4absmax, compare4absmin

    \ingroup libtensor_tod
 **/
template<size_t N, typename ComparePolicy = compare4absmin>
class tod_select : public noncopyable {
public:
    typedef ComparePolicy compare_t;

    typedef tensor_element<N, double> tensor_element_type;
    typedef std::list<tensor_element_type> list_type; //!< List type for index-value pairs

private:
    dense_tensor_rd_i<N, double> &m_t; //!< Tensor
    permutation<N> m_perm; //!< Permutation of tensor
    double m_c; //!< Scaling coefficient
    compare_t m_cmp; //!< Compare policy object to select entries

public:
    /** \brief Constuctor
        \param t Tensor.
        \param tr Tensor transformation
        \param cmp Compare policy.
    **/
    tod_select(dense_tensor_rd_i<N, double> &t,
            const tensor_transf<N, double> &tr,
            compare_t cmp = compare_t()) :
        m_t(t), m_perm(tr.get_perm()),
        m_c(tr.get_scalar_tr().get_coeff()), m_cmp(cmp)
    { }

    /** \brief Constuctor
        \param t Tensor.
        \param cmp Compare policy.
    **/
    tod_select(dense_tensor_rd_i<N, double> &t, compare_t cmp = compare_t()) :
        m_t(t), m_c(1.0), m_cmp(cmp)
    { }

    /** \brief Constuctor
        \param t Tensor.
        \param c Coefficient.
        \param cmp Compare policy.
    **/
    tod_select(dense_tensor_rd_i<N, double> &t, double c,
        compare_t cmp = compare_t()) :
        m_t(t), m_c(c), m_cmp(cmp)
    { }

    /** \brief Constuctor
        \param t Tensor
        \param p Permutation
        \param c Coefficient
        \param cmp Compare policy object.
    **/
    tod_select(dense_tensor_rd_i<N, double> &t, const permutation<N> &p,
        double c, compare_t cmp = compare_t()) :
        m_t(t), m_perm(p), m_c(c), m_cmp(cmp)
    { }

    /** \brief Selects the index-value pairs from the tensor
        \param li List of index-value pairs.
        \param n Maximum size of the list.
    **/
    void perform(list_type &li, size_t n);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SELECT_H

#include "impl/tod_select_impl.h"
