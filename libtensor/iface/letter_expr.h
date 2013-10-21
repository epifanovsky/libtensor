#ifndef LIBTENSOR_LETTER_EXPR_H
#define LIBTENSOR_LETTER_EXPR_H

#include <vector>
#include <libtensor/exception.h>
#include <libtensor/core/permutation_builder.h>
#include "letter.h"

/** \defgroup libtensor_letter_expr Letter index expressions
    \ingroup libtensor_iface

    The members of this group provide the facility to operate %letter
    indexes.

    <b>See also:</b>

     * libtensor::letter
**/

namespace libtensor {


/** \brief Base class for %letter %index expressions

    \ingroup libtensor_letter_expr
**/
template<size_t N>
class letter_expr_base {
};

/** \brief Expression using %letter %tensor indexes

    \ingroup libtensor_letter_expr
**/
template<size_t N>
class letter_expr : public letter_expr_base<N> {
private:
    letter_expr<N - 1> m_expr;
    const letter &m_let;

public:
    letter_expr(const letter_expr<N - 1> &expr, const letter &let) :
        m_expr(expr), m_let(let) { }
    letter_expr(const std::vector<const letter*> &v) :
        m_expr(v), m_let(*v[N - 1]) { }
    letter_expr(const letter_expr<N> &expr) :
        m_expr(expr.m_expr), m_let(expr.m_let) { }

    /** \brief Returns whether the expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return &m_let == &let || m_expr.contains(let);
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const throw(exception) {
        if(&m_let == &let) return N - 1;
        return m_expr.index_of(let);
    }

    /** \brief Returns the %letter at a given position
        \throw out_of_bounds If the %index is out of bounds.
     **/
    const letter &letter_at(size_t i) const throw(out_of_bounds) {
        if(i == N - 1) return m_let;
        return m_expr.letter_at(i);
    }

    /** \brief Returns how letters in the second expression need to be
            permuted to obtain the order of letters in this
            expression
        \param e2 Second expression.
     **/
    permutation<N> permutation_of(const letter_expr<N> &expr) const
        throw(exception) {

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
    letter_expr<N> &operator=(const letter_expr<N> &);
};

template<>
class letter_expr<1> : public letter_expr_base<1> {
private:
    const letter &m_let;

public:
    letter_expr(const letter &let) : m_let(let) { }
    letter_expr(const std::vector<const letter*> &v) : m_let(*v[0]) { }
    letter_expr(const letter_expr<1> &expr) : m_let(expr.m_let) { }

    /** \brief Returns whether the expression contains a %letter
     **/
    bool contains(const letter &let) const {
        return &m_let == &let;
    }

    /** \brief Returns the %index of a %letter in the expression
        \throw exception If the expression doesn't contain the %letter.
     **/
    size_t index_of(const letter &let) const throw(exception) {
        if(&m_let != &let) {
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

    template<size_t M>
    void unfold(const letter *(&seq)[M]) const {
        seq[0] = &m_let;
    }

private:
    letter_expr<1> &operator=(const letter_expr<1> &);
};


template<> class letter_expr<0>;


/** \brief Bitwise OR (|) operator for two letters

    \ingroup libtensor_letter_expr
**/
inline letter_expr<2> operator|(const letter &l1, const letter &l2) {
    if(&l1 == &l2) {
        throw_exc("", "operator|(const letter&, const letter&)",
            "Only unique letters are allowed");
    }
    return letter_expr<2>(letter_expr<1>(l1), l2);
}


/** \brief Bitwise OR (|) operator for an expression and a %letter

    \ingroup libtensor_letter_expr
**/
template<size_t N>
inline letter_expr<N + 1> operator|(letter_expr<N> expr1, const letter &l2) {
    if(expr1.contains(l2)) {
        throw_exc("", "operator|(letter_expr, const letter&)",
            "Only unique letters are allowed");
    }
    return letter_expr<N + 1>(expr1, l2);
}


/** \brief Create permutation between two letter_expr
 **/
template<size_t N>
permutation<N> match(const letter_expr<N> &e1, const letter_expr<N> &e2) {

    sequence<N, size_t> seq1(0), seq2(0);
    for (size_t i = 0; i < N; i++) {
        const letter &l = e1.letter_at(i);
        if (! e2.contains(l)) {
            throw_exc("", "match(const letter_expr<N> &, "
                    "const letter_expr<N> &e2", "Letter not found.");
        }
        seq1[i] = i;
        seq2[i] = e2.index_of(l);
    }

    return permutation_builder<N>(seq1, seq2).get_perm();
}

} // namespace libtensor

#endif // LIBTENSOR_LETTER_EXPR_H

