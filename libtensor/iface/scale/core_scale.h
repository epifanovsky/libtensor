#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T> class eval_scale;


/** \brief Expression core that scales an underlying expression
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T>
class core_scale : public expr_core_i<N, T> {
public:
    static const char k_clazz[]; //!< Class name

public:
    //!    Evaluating container type
    typedef eval_scale<N, T> eval_container_t;

private:
    T m_coeff; //!< Scaling coefficient
    expr<N, T> m_expr; //!< Unscaled expression

public:
    /** \brief Constructs the scaling expression using a coefficient
            and the underlying unscaled expression
     **/
    core_scale(const T &coeff, const expr<N, T> &subexpr) :
        m_coeff(coeff), m_expr(subexpr)
    { }

    /** \brief Returns the unscaled expression
     **/
    expr<N, T> &get_unscaled_expr() {
        return m_expr;
    }

    /** \brief Returns the unscaled expression (const version)
     **/
    const expr<N, T> &get_unscaled_expr() const {
        return m_expr;
    }

    /** \brief Returns the scaling coefficient
     **/
    const T &get_coeff() {
        return m_coeff;
    }

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr.letter_at(i);
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_SCALE_H
