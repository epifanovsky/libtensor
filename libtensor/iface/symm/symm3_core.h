#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, bool Sym, typename T> class symm3_eval;


/** \brief Expression core for the symmetrization over three indexes
    \tparam N Tensor order.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, bool Sym, typename T>
class symm3_core {
public:
    static const char k_clazz[]; //!< Class name

public:
     //!    Evaluating container type
    typedef symm3_eval<N, Sym, T, SubCore> eval_container_t;

private:
    const letter &m_l1; //!< First %index
    const letter &m_l2; //!< Second %index
    const letter &m_l3; //!< Third %index
    expr<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param l1 First symmetrized %index.
        \param l2 Second symmetrized %index.
        \param l3 Third symmetrized %index.
        \param subexpr Sub-expression.
     **/
    symm3_core(const letter &l1, const letter &l2, const letter &l3,
        const expr<N, T> &subexpr);

    /** \brief Returns the first symmetrized index
     **/
    const letter &get_l1() const {
        return m_l1;
    }

    /** \brief Returns the second symmetrized index
     **/
    const letter &get_l2() const {
        return m_l2;
    }

    /** \brief Returns the third symmetrized index
     **/
    const letter &get_l3() const {
        return m_l3;
    }

    /** \brief Returns the sub-expression
     **/
    expr<N, T> &get_sub_expr() {
        return m_expr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr<N, T> &get_sub_expr() const {
        return m_expr;
    }

    /** \brief Returns whether the result's label contains a letter
        \param let Letter.
     **/
    bool contains(const letter &let) const {
        return m_expr.contains(let);
    }

    /** \brief Returns the index of a letter in the result's label
        \param let Letter.
        \throw expr_exception If the label does not contain the
            requested letter.
     **/
    size_t index_of(const letter &let) const {
        return m_expr.index_of(let);
    }

    /** \brief Returns the letter at a given position in the result's label
        \param i Letter index.
        \throw out_of_bounds If the index is out of bounds.
     **/
    const letter &letter_at(size_t i) const {
        return m_expr.letter_at(i);
    }

};


template<size_t N, bool Sym, typename T>
const char symm3_core<N, Sym, T>::k_clazz[] = "symm3_core<N, Sym, T>";


template<size_t N, bool Sym, typename T>
symm3_core<N, Sym, T>::symm3_core(const letter &l1, const letter &l2,
    const letter &l3, const expr<N, T> &subexpr) :

    m_l1(l1), m_l2(l2), m_l3(l3), m_subexpr(subexpr) {

    static const char method[] = "symm3_core(const letter&, "
        "const letter&, const letter&, const expr<N, T>&)";

    if(m_l1 == m_l2 || m_l1 == m_l3 || m_l2 == m_l3) {
        throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
            "Symmetrized indexes must be different.");
    }
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM3_CORE_H
