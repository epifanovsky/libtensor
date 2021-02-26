#ifndef LIBTENSOR_EXPR_LABEL_H
#define LIBTENSOR_EXPR_LABEL_H

#include <vector>
#include <libtensor/exception.h>
#include <libtensor/core/permutation_builder.h>
#include "letter.h"

namespace libtensor {
namespace expr {


/** \brief Expression using %letter %tensor indexes

    \ingroup libtensor_expr_iface
**/
template<size_t N>
class label {
private:
    label<N - 1> m_expr;
    const letter &m_let;

public:
    label(const label<N - 1> &expr, const letter &let) :
        m_expr(expr), m_let(let) { }
    label(const std::vector<const letter*> &v) :
        m_expr(v), m_let(*v[N - 1]) { }
    label(const label<N> &expr) :
        m_expr(expr.m_expr), m_let(expr.m_let) { }

    /** \brief Returns whether the expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return &m_let == &let || m_expr.contains(let);
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const {
        if(&m_let == &let) return N - 1;
        return m_expr.index_of(let);
    }

    /** \brief Returns the %letter at a given position
        \throw out_of_bounds If the %index is out of bounds.
     **/
    const letter &letter_at(size_t i) const {
        if(i == N - 1) return m_let;
        return m_expr.letter_at(i);
    }

    /** \brief Returns how letters in the second expression need to be
            permuted to obtain the order of letters in this
            expression
        \param e2 Second expression.
     **/
    permutation<N> permutation_of(const label<N> &expr) const {

        const letter *seq1[N], *seq2[N];
        unfold(seq1);
        expr.unfold(seq2);
        permutation_builder<N> pb(seq1, seq2);
        return permutation<N>(pb.get_perm());
    }

    template<size_t M>
    void unfold(const letter *(&seq)[M]) const {
        m_expr.unfold(seq);
        seq[N - 1] = &m_let;
    }

private:
    label<N> &operator=(const label<N> &);
};


template<>
class label<1> {
private:
    const letter &m_let;

public:
    label(const letter &let) : m_let(let) { }
    label(const std::vector<const letter*> &v) : m_let(*v[0]) { }
    label(const label<1> &expr) : m_let(expr.m_let) { }

    /** \brief Returns whether the expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return &m_let == &let;
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const {
        if(&m_let != &let) {
            throw_exc("letter_expr<1>", "index_of()",
                "Expression doesn't contain the letter.");
        }
        return 0;
    }

    /** \brief Returns the %letter at a given position
        \throw exception If the %index is out of bounds.
     **/
    const letter &letter_at(size_t i) const {
        if(i != 0) {
            throw out_of_bounds(g_ns, "letter_expr<1>",
                "letter_at(size_t)", __FILE__, __LINE__,
                "Letter index is out of bounds.");
        }
        return m_let;
    }

    permutation<1> permutation_of(const label<1> &e2) const {
        return permutation<1>();
    }

    template<size_t M>
    void unfold(const letter *(&seq)[M]) const {
        seq[0] = &m_let;
    }

private:
    label<1> &operator=(const label<1> &);
};


template<> class label<0>;


/** \brief Bitwise OR (|) operator for two letters

    \ingroup libtensor_letter_expr
**/
inline label<2> operator|(const letter &l1, const letter &l2) {
    if(&l1 == &l2) {
        throw_exc("", "operator|(const letter&, const letter&)",
            "Only unique letters are allowed");
    }
    return label<2>(label<1>(l1), l2);
}


/** \brief Bitwise OR (|) operator for an expression and a %letter

    \ingroup libtensor_letter_expr
**/
template<size_t N>
inline label<N + 1> operator|(label<N> expr1, const letter &l2) {
    if(expr1.contains(l2)) {
        throw_exc("", "operator|(letter_expr, const letter&)",
            "Only unique letters are allowed");
    }
    return label<N + 1>(expr1, l2);
}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_LABEL_H
