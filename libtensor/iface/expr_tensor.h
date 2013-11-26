#ifndef LIBTENSOR_IFACE_EXPR_TENSOR_H
#define LIBTENSOR_IFACE_EXPR_TENSOR_H

#include <memory>
#include <libtensor/core/noncopyable.h>
#include "any_tensor.h"
#include "labeled_lhs_rhs.h"

namespace libtensor {
namespace iface {


/** \brief Tensor-like object that stores a tensor expression

    \ingroup libtensor_iface
 **/
template<size_t N, typename T = double>
class expr_tensor :
    public any_tensor<N, T>, public expr_lhs<N, T>, public noncopyable {

private:
    expr::expr_tree *m_expr; //!< Expression

public:
    /** \brief Constructs an empty object
     **/
    expr_tensor() : m_expr(0) { }

    /** \brief Virtual destructor
     **/
    virtual ~expr_tensor() { }

    /** \brief Attaches a letter label to expr_tensor
     **/
    labeled_lhs_rhs<N, T> operator()(const letter_expr<N> &label);

    /** \brief Saves the given expression in this container
     **/
    virtual void assign(const expr_rhs<N, T> &rhs, const letter_expr<N> &label);

protected:
    /** \brief Redefined any_tensor::make_rhs
     **/
    virtual expr_rhs<N, T> make_rhs(const letter_expr<N> &label);

};


template<size_t N, typename T>
labeled_lhs_rhs<N, T> expr_tensor<N, T>::operator()(
    const letter_expr<N> &label) {

    return labeled_lhs_rhs<N, T>(*this, label, make_rhs(label));
}


template<size_t N, typename T>
void expr_tensor<N, T>::assign(const expr_rhs<N, T> &rhs,
    const letter_expr<N> &label) {

//    m_expr.reset(rhs.get_core().clone());
}


template<size_t N, typename T>
expr_rhs<N, T> expr_tensor<N, T>::make_rhs(const letter_expr<N> &label) {

//    return expr_rhs<N, T>(*m_expr);
}


} // namespace iface
} // namespace libtensor


namespace libtensor {

using iface::expr_tensor;

} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_TENSOR_H
