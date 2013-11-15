#ifndef LIBTENSOR_EXPR_NODE_H
#define LIBTENSOR_EXPR_NODE_H

#include <vector>
#include <string>

namespace libtensor {
namespace expr {


/** \brief Basic node of the expression tree

    \ingroup libtensor_expr
 **/
class node {
private:
    std::string m_op; //!< Operation name
    size_t m_n; //!< Order of tensor represented by node

public:
    /** \brief Creates this node
        \param n Order of result.
     **/
    node(const std::string &op, size_t n) : m_op(op), m_n(n) { }

    /** \brief Virtual destructor
     **/
    virtual ~node() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Returns the operation
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
