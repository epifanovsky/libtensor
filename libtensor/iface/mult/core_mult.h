#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, typename T, bool Recip> class eval_mult;


/** \brief Element-wise multiplication operation expression core
    \tparam N Tensor order.
    \tparam T Tensor element type.
    \tparam Recip If true do element-wise division instead of multiplication.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, typename T, bool Recip>
class core_mult : public expr_core_i<N, T> {
public:
    //! Evaluating container type
    typedef eval_mult<N, T, Recip> eval_container_t;

private:
    expr<N, T> m_expr1; //!< Left expression
    expr<N, T> m_expr2; //!< Right expression

public:
    //!    \name Construction
    //@{

    /** \brief Initializes the core with left and right expressions
     **/
    core_mult(const expr<N, T> &expr1, const expr<N, T> &expr2) :
        m_expr1(expr1), m_expr2(expr2)
    { }

    //@}

    /** \brief Returns the first expression
     **/
    expr<N, T> &get_expr_1() {
        return m_expr1;
    }

    /** \brief Returns the first expression (const version)
     **/
    const expr<N, T> &get_expr_1() const {
        return m_expr1;
    }

    /** \brief Returns the second expression
     **/
    expr<N, T> &get_expr_2() {
        return m_expr2;
    }

    /** \brief Returns the second expression (const version)
     **/
    const expr<N, T> &get_expr_2() const {
        return m_expr2;
    }

    /** \brief Returns whether the tensor's label contains a letter
     **/
    bool contains(const letter &let) const {
        return m_expr1.contains(let);
    }

    /** \brief Returns the index of a letter in the tensor's label
     **/
    size_t index_of(const letter &let) const {
        return m_expr1.index_of(let);
    }

    /** \brief Returns the letter at a given position in the tensor's label
     **/
    const letter &letter_at(size_t i) const {
        return m_expr1.letter_at(i);
    }

};


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_CORE_MULT_H
