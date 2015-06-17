#ifndef NODE_REBLOCK_H
#define NODE_REBLOCK_H

#include "node.h"

namespace libtensor {
namespace expr {

/** \brief Tensor expression node: reblocking

    \ingroup libtensor_expr_dag
 **/
class node_reblock : public node {
public:
    static const char k_op_type[]; //!< Operation type

private:
    size_t m_subspace;
public:
    /** \brief Creates an reblock node
        \param n Order of result.
     **/
    node_reblock(size_t n,size_t subspace) :
        node(k_op_type, n),
        m_subspace(subspace)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_reblock() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_reblock(*this);
    }

    size_t get_subspace() const { return m_subspace; }

};


} // namespace expr
} // namespace libtensor


#endif /* NODE_REBLOCK_H */
