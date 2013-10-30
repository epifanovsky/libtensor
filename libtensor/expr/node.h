#ifndef LIBTENSOR_EXPR_NODE_H
#define LIBTENSOR_EXPR_NODE_H

#include <map>
#include <string>

namespace libtensor {
namespace expr {


/** \brief Basic node of the expression tree

    \ingroup libtensor_expr
 **/
class node {
public:
    typedef size_t tid_t; //!< Tensor ID type

private:
    std::string m_op; //!< Operation

public:
    /** \brief Creates this node
        \param op Operation name.
     **/
    node(const std::string &op) :
        m_op(op)
    { }

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
