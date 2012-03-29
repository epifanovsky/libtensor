#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H

#include "../../defs.h"
#include "../../exception.h"
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, typename Expr> class eval_scale;


/** \brief Expression core that scales an underlying expression
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Expr Underlying expression type (labeled_btensor_expr).

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, typename Expr>
class core_scale {
public:
    static const char *k_clazz; //!< Class name

public:
    //!    Evaluating container type
    typedef eval_scale<N, T, Expr> eval_container_t;

private:
    Expr m_expr; //!< Unscaled expression
    T m_coeff; //!< Scaling coefficient

public:
    /** \brief Constructs the scaling expression using a coefficient
            and the underlying unscaled expression
     **/
    core_scale(T coeff, const Expr &expr) : m_coeff(coeff), m_expr(expr) { }

    /** \brief Returns the unscaled expression
     **/
    Expr &get_unscaled_expr() { return m_expr; }

    /** \brief Returns the scaling coefficient
     **/
    T get_coeff() { return m_coeff; }

    /** \brief Returns whether the %tensor's label contains a %letter
     **/
    bool contains(const letter &let) const;

    /** \brief Returns the %index of a %letter in the %tensor's label
     **/
    size_t index_of(const letter &let) const throw(exception);

    /** \brief Returns the %letter at a given position in
            the %tensor's label
     **/
    const letter &letter_at(size_t i) const throw(exception);

};


template<size_t N, typename T, typename Expr>
const char *core_scale<N, T, Expr>::k_clazz = "core_scale<N, T, Expr>";


template<size_t N, typename T, typename Expr>
inline bool core_scale<N, T, Expr>::contains(const letter &let) const {

    return m_expr.contains(let);
}


template<size_t N, typename T, typename Expr>
inline size_t core_scale<N, T, Expr>::index_of(const letter &let) const
    throw(exception) {

    return m_expr.index_of(let);
}


template<size_t N, typename T, typename Expr>
inline const letter &core_scale<N, T, Expr>::letter_at(size_t i) const
    throw(exception) {

    return m_expr.letter_at(i);
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H
