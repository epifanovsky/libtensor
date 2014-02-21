#ifndef LIBTENSOR_EXPR_NODE_SYMM_H
#define LIBTENSOR_EXPR_NODE_SYMM_H

#include <vector>
#include <libtensor/core/scalar_transf.h>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: tensor symmetrization (base class)

    The symmetrization sequence describes the n-fold symmetrization of the
    node. Elements (1,...,n), (n+1,...,2n), and so on of the sequence refer to
    groups of indexes. The indexes of each group are symmetrized among
    themselves and at the same time as the other groups. I.e. all groups have
    to have the same size.

    \ingroup libtensor_expr_dag
 **/
class node_symm_base : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::vector<size_t> m_sym; //!< Symmetrization sequence
    size_t m_nsym; //!< Order of symmetrization (2: pair sym, 3: triple sym, ..)

public:
    /** \brief Creates an identity node
        \param n Order of result.
        \param sym Symmetrization sequence.
        \param nsym Order of symmetrization.
     **/
    node_symm_base(size_t n, const std::vector<size_t> &sym, size_t nsym) :
        node(k_op_type, n), m_sym(sym), m_nsym(nsym)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_symm_base() { }

    /** \brief Returns the symmetrization sequence
     **/
    const std::vector<size_t> &get_sym() const {
        return m_sym;
    }

    /** \brief Returns the symmetrization order
     **/
    size_t get_nsym() const { return m_nsym; }

};


/** \brief Expression node: tensor symmetrization

    \ingroup libtensor_expr_dag
 **/
template<typename T>
class node_symm : public node_symm_base {
private:
    scalar_transf<T> m_trp, m_trc; //!< Pair and cyclic transform

public:
    /** \brief Creates an symmetrization node
     **/
    node_symm(
        size_t n,
        const std::vector<size_t> &sym,
        size_t nsym,
        const scalar_transf<T> &trp,
        const scalar_transf<T> &trc) :
        node_symm_base(n, sym, nsym), m_trp(trp), m_trc(trc)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_symm() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_symm<T>(*this);
    }

    /** \brief Returns the pair scalar transform
     **/
    const scalar_transf<T> &get_pair_tr() const { return m_trp; }

    /** \brief Returns the cyclic scalar transform
     **/
    const scalar_transf<T> &get_cyclic_tr() const { return m_trc; }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SYMM_BASE_H
