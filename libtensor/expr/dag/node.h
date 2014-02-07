#ifndef LIBTENSOR_EXPR_NODE_H
#define LIBTENSOR_EXPR_NODE_H

#include <cstdlib> // for size_t
#include <string>

namespace libtensor {
namespace expr {


/** \brief Basic tensor expression node

    Base class for nodes in the directed acyclic graph (dag) of an expression.
    Each node in the graph represents a single operation and is connected to
    the arguments by graph edges.

    This base class provides some core functionality and does not represent
    any particular operation. Derived classes representing concrete operations
    shall provide the name of the operation as well as the order of the tensor
    that results from that operation.

    \sa dag

    \ingroup libtensor_expr_dag
 **/
class node {
private:
    std::string m_op; //!< Name of tensor operation
    size_t m_n; //!< Order of tensor represented by node

public:
    /** \brief Constructor of basic node
        \param op Name of operation.
        \param n Order of result.
     **/
    node(const std::string &op, size_t n) :
        m_op(op), m_n(n)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node() { }

    /** \brief Creates an exact copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the name of the operation
     **/
    const std::string &get_op() const {
        return m_op;
    }

    /** \brief Returns the order of the result
     **/
    size_t get_n() const {
        return m_n;
    }

    /** \brief Dynamically recasts this node onto a derived type
        \tparam T Derived node type.
     **/
    template<typename T>
    const T &recast_as() const {
        return dynamic_cast<const T&>(*this);
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_H
