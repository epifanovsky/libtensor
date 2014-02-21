#ifndef LIBTENSOR_EXPR_BISPACE_EXPR_H
#define LIBTENSOR_EXPR_BISPACE_EXPR_H

#include <libtensor/core/permutation_builder.h>
#include "bispace.h"

namespace libtensor {
namespace expr {

template<size_t N> class bispace;

namespace bispace_expr {


/** \brief Block %index space expression
    \tparam N Expression order.
    \tparam C Expression core type.

    \ingroup libtensor_bispace_expr
 **/
template<size_t N, typename C>
class expr {
public:
    typedef C core_t; //!< Expression core type

private:
    core_t m_core; //!< Expression core

public:
    /** \brief Creates the expression using a core object
     **/
    expr(const core_t &core) : m_core(core) { }

    /** \brief Copy constructor
     **/
    expr(const expr<N, C> &e) : m_core(e.m_core) { }

    /** \brief Returns the core of the expression
     **/
    const core_t &get_core() const {
        return m_core;
    }

    /** \brief Compares two expressions
     **/
    bool equals(const expr<N, C> &other) const {
        return m_core.equals(other.m_core);
    }

    /** \brief Returns a single-dimension subspace
     **/
    const bispace<1> &at(size_t i) const {
        return m_core.at(i);
    }

    /** \brief Returns the number of times this expression contains
            a subexpression
        \tparam M Order of the subexpression.
        \tparam D Expression core type of the subexpression.
     **/
    template<size_t M, typename D>
    size_t contains(const expr<M, D> &subexpr) const {
        return m_core.contains(subexpr);
    }

    /** \brief Returns the first location where the subexpression is
            found
        \tparam M Order of the subexpression.
        \tparam D Expression core type of the subexpression.
        \throw expr_exception If the subexpression cannot be found.
     **/
    template<size_t M, typename D>
    size_t locate(const expr<M, D> &subexpr) const {
        return m_core.locate(subexpr);
    }

    /** \brief Builds a permutation of all single-dimension subspaces
            in another expression with respect to this one
     */
    template<typename D>
    void build_permutation(
        const expr<N, D> &other, permutation<N> &perm) const {
        size_t seq1[N], seq2[N];
        for(size_t i = 0; i < N; i++) seq1[i] = i;
        record_pos(other, 0, seq2);
        permutation_builder<N> pb(seq2, seq1);
        perm.permute(pb.get_perm());
    }

    template<size_t M, typename D>
    void record_pos(const expr<M, D> &supexpr, size_t pos_here,
        size_t (&perm)[M]) const {

        m_core.record_pos(supexpr, pos_here, perm);
    }

    void mark_sym(size_t i, mask<N> &msk) const {
        m_core.mark_sym(i, msk, 0);
    }

    template<size_t M>
    void mark_sym(size_t i, mask<M> &msk, size_t offs) const {
        m_core.mark_sym(i, msk, offs);
    }

};


} // namespace bispace_expr
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_BISPACE_EXPR_H
