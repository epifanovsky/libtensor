#ifndef LIBTENSOR_LETTER_EXPR_H
#define LIBTENSOR_LETTER_EXPR_H

#include "../exception.h"
#include <libtensor/core/sequence.h>
#include "../core/permutation_builder.h"
#include "letter.h"

namespace libtensor {


/** \brief Sequence of letters used to label tensor indexes

    \ingroup libtensor_expr
 **/
template<size_t N>
class letter_expr {
private:
    sequence<N, const letter*> m_expr; //!< Sequence of letters

public:
    /** \brief Constructs the expression by appending a letter
     **/
    letter_expr(const letter_expr<N - 1> &expr, const letter &let) : m_expr(0) {

        for(size_t i = 0; i < N - 1; i++) m_expr[i] = &expr.letter_at(i);
        m_expr[N - 1] = &let;
    }

    /** \brief Constructs the expression by appending another letter_expr
     **/
    template<size_t K>
    letter_expr(const letter_expr<K> &expr1, const letter_expr<N - K> &expr2) :
        m_expr(0) {

        for(size_t i = 0; i < K; i++) m_expr[i] = &expr1.letter_at(i);
        for(size_t i = K; i < N; i++) m_expr[i] = &expr2.letter_at(i - K);
    }

    /** \brief Constructs the expression from a sequence of letters
     **/
    letter_expr(const sequence<N, const letter*> &seq) : m_expr(seq) { }

    /** \brief Returns whether the expression contains the given letter
     **/
    bool contains(const letter &let) const {

        for(size_t i = 0; i < N; i++) if(let == *m_expr[i]) return true;
        return false;
    }

    /** \brief Returns the index of a letter in the expression
            or N if the letter is not found
     **/
    size_t index_of(const letter &let) const {

        for(size_t i = 0; i < N; i++) if(let == *m_expr[i]) return i;
        return N;
    }

    /** \brief Returns the letter at a given position
     **/
    const letter &letter_at(size_t i) const {

        return *m_expr[i];
    }

    /**	\brief Returns how letters in the second expression need to be
            permuted to obtain the order of letters in this expression
        \param expr Second expression.
     **/
    permutation<N> permutation_of(const letter_expr<N> &expr) const {

        return permutation_builder<N>(m_expr, expr.m_expr).get_perm();
    }

};


/** \brief Single-letter expression specialization

    \ingroup libtensor_expr
 **/
template<>
class letter_expr<1> {
private:
    const letter &m_let;

public:
    letter_expr(const letter &let) : m_let(let) { }
    letter_expr(const letter_expr<1> &expr) : m_let(expr.m_let) { }
    letter_expr(const sequence<1, const letter*> &seq) : m_let(*seq[0]) { }

    /** \brief Returns whether the expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return m_let == let;
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const throw(exception) {
        if(m_let != let) {
            throw_exc("letter_expr<1>", "index_of()",
                    "Expression doesn't contain the letter.");
        }
        return 0;
    }

    /** \brief Returns the %letter at a given position
        \throw exception If the %index is out of bounds.
     **/
    const letter &letter_at(size_t i) const throw(out_of_bounds) {
        if(i != 0) {
            throw out_of_bounds(g_ns, "letter_expr<1>",
                "letter_at(size_t)", __FILE__, __LINE__,
                "Letter index is out of bounds.");
        }
        return m_let;
    }

    permutation<1> permutation_of(const letter_expr<1> &e2) const {
        return permutation<1>();
    }

};


/** \brief Bitwise OR (|) operator for two letters

    \ingroup libtensor_expr
 **/
inline letter_expr<2> operator|(const letter &l1, const letter &l2) {

    return letter_expr<2>(letter_expr<1>(l1), l2);
}


/** \brief Bitwise OR (|) operator for an expression and a %letter

    \ingroup libtensor_expr
 **/
template<size_t N>
inline letter_expr<N + 1> operator|(const letter_expr<N> &expr1,
    const letter &l2) {

    return letter_expr<N + 1>(expr1, l2);
}


} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_H
