#ifndef LIBTENSOR_EXPR_NODE_SCALAR_H
#define LIBTENSOR_EXPR_NODE_SCALAR_H

#include <typeinfo>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: scalar container (base class)

    \ingroup libtensor_expr_dag
 **/
class node_scalar_base : public node {
public:
    static const char k_op_type[];

public:
    /** \brief Creates the node
     **/
    node_scalar_base() :
        node(k_op_type, 0)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_scalar_base() { }

    /** \brief Returns the type of scalar
     **/
    virtual const std::type_info &get_type() const = 0;

};


/** \brief Expression node: scalar container
    \tparam T Scalar type.

    This node contains a reference to a scalar.

    \ingroup libtensor_expr_dag
 **/
template<typename T>
class node_scalar : public node_scalar_base {
private:
    T &m_scalar; //!< Reference to scalar

public:
    /** \brief Creates the node
        \param a Reference to scalar.
     **/
    node_scalar(T &a) :
        m_scalar(a)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_scalar() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_scalar<T>(*this);
    }

    /** \brief Returns the type of scalar
     **/
    virtual const std::type_info &get_type() const {
        return typeid(T);
    }

    /** \brief Returns the reference to the scalar
     **/
    T &get_scalar() const {
        return m_scalar;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SCALAR_H
