#ifndef LIBTENSOR_IFACE_EXPR_RHS_H
#define LIBTENSOR_IFACE_EXPR_RHS_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/expr/dag/expr_tree.h>
#include "letter_expr.h"


namespace libtensor {
namespace iface {


/** \brief Expression meta-wrapper
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Core Expression core type.

    Tensor expressions make extensive use of a meta-programming technique
    call "expression templates". It allows us to store the expression
    tree as the C++ type thus transferring a number of sanity checks to
    the compilation level.

    This template wraps around the real expression type (core) to facilitate
    the matching of overloaded operator templates.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class expr_rhs : public noncopyable {
private:
    expr::expr_tree m_expr; //!< Expression
    letter_expr<N> m_label; //!< Letter label

public:
    /** \brief Constructs the expression using a core
     **/
    expr_rhs(const expr::expr_tree &expr, const letter_expr<N> &l) :
        m_expr(expr), m_label(l) { }

    /** \brief Copy constructor
     **/
    expr_rhs(const expr_rhs<N, T> &expr) :
        m_expr(expr.m_expr), m_label(expr.m_label) { }

    /** \brief Virtual destructor
     **/
    virtual ~expr_rhs() { }

    /** \brief Returns the core of the expression
     **/
    expr::expr_tree &get_expr() {
        return m_expr;
    }

    /** \brief Returns the core of the expression (const version)
     **/
    const expr::expr_tree &get_expr() const {
        return m_expr;
    }

    const letter_expr<N> &get_label() const {
        return m_label;
    }

    /** \brief Returns whether the label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_label.contains(let);
    }

    /** \brief Returns the %index of a %letter in the label
     **/
    size_t index_of(const letter &let) const {
        return m_label.index_of(let);
    }

    /** \brief Returns the %letter at a given position in the label
     **/
    const letter &letter_at(size_t i) const {
        return m_label.letter_at(i);
    }
};


} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_IFACE_EXPR_RHS_H
