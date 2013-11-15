#ifndef LIBTENSOR_EXPR_NODE_SYMM_BASE_H
#define LIBTENSOR_EXPR_NODE_SYMM_BASE_H

#include <vector>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Tensor symmetrization node of the expression tree

    The symmetrization sequence describes the n-fold symmetrization of the
    node. Elements (1,...,n), (n+1,...,2n), and so on of the sequence refer to
    groups of indexes. The indexes of each group are symmetrized among
    themselves and at the same time as the other groups. I.e. all groups have
    to have the same size.

    \ingroup libtensor_expr
 **/
class node_symm_base : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    std::vector<size_t> m_sym; //!< Symmetrization sequence
    size_t m_nsym; //!< Order of symmetrization (2: pair sym, 3: triple sym, ..)

public:
    /** \brief Creates an identity node
        \param n Order of result
        \param sym Symmetrization sequence
        \param nsym Order of symmetrization
     **/
    node_symm_base(size_t n, const std::vector<size_t> &sym, size_t nsym) :
        node(node_symm_base::k_op_type, n), m_sym(sym), m_nsym(nsym)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_symm_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_symm_base *clone() const {
        return new node_symm_base(*this);
    }

    /** \brief Returns the symmetrization sequence
     **/
    const std::vector<size_t> &get_sym() const {
        return m_sym;
    }

    size_t get_nsym() const { return m_nsym; }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SYMM_BASE_H
