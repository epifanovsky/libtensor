#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H

#include <libtensor/exception.h>
#include "../expr_exception.h"
#include "../letter.h"
#include "../letter_expr.h"
#include "../labeled_btensor_expr.h" // for g_ns

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N, size_t M, bool Sym, typename T> class symm2_eval;


/** \brief Expression core for the symmetrization over two sets of indexes
    \tparam N Tensor order.
    \tparam M Number of indexes in the set.
    \tparam Sym Symmetrization/antisymmetrization.

    \ingroup libtensor_btensor_expr
 **/
template<size_t N, size_t M, bool Sym, typename T>
class symm2_core {
public:
    static const char k_clazz[]; //!< Class name

public:
     //!    Evaluating container type
    typedef symm2_eval<N, M, Sym, T> eval_container_t;

private:
    letter_expr<M> m_sym1; //!< First set of symmetrized indexes
    letter_expr<M> m_sym2; //!< Second set of symmetrized indexes
    expr<N, T> m_subexpr; //!< Sub-expression

public:
    /** \brief Creates the expression core
        \param sym1 First expression indicating symmetrized indexes
        \param sym2 Second expression indicating symmetrized indexes
        \param subexpr Sub-expression.
     **/
    symm2_core(const letter_expr<M> &sym1, const letter_expr<M> &sym2,
        const expr<N, T> &subexpr);

    /** \brief Returns the first set of symmetrized indexes
     **/
    const letter_expr<M> &get_sym1() const {
        return m_sym1;
    }

    /** \brief Returns the second set of symmetrized indexes
     **/
    const letter_expr<M> &get_sym2() const {
        return m_sym2;
    }

    /** \brief Returns the sub-expression
     **/
    expr<N, T> &get_sub_expr() {
        return m_subexpr;
    }

    /** \brief Returns the sub-expression, const version
     **/
    const expr<N, T> &get_sub_expr() const {
        return m_subexpr;
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


template<size_t N, size_t M, bool Sym, typename T>
const char symm2_core<N, M, Sym, T>::k_clazz[] = "symm2_core<N, M, Sym, T>";


template<size_t N, size_t M, bool Sym, typename T>
symm2_core<N, M, Sym, T>::symm2_core(const letter_expr<M> &sym1,
    const letter_expr<M> &sym2, const expr<N, T> &expr) :

    m_sym1(sym1), m_sym2(sym2), m_expr(expr) {

    static const char method[] = "symm2_core(const letter_expr<M>&, "
        "const letter_expr<M>&, const expr<N, T>&)";

    for(size_t i = 0; i < M; i++) {
        if(sym2.contains(sym1.letter_at(i))) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Symmetrized indexes must be different.");
        }
    }
    for(size_t i = 0; i < M; i++) {
        const letter &l1 = m_sym1.letter_at(i);
        const letter &l2 = m_sym2.letter_at(i);
        if(!m_expr.contains(l1) || !m_expr.contains(l2)) {
            throw expr_exception(g_ns, k_clazz, method, __FILE__, __LINE__,
                "Symmetrized index is absent from the sub-expression.");
        }
    }
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_SYMM2_CORE_H
