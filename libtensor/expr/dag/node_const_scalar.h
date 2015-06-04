#ifndef LIBTENSOR_EXPR_NODE_CONST_SCALAR_H
#define LIBTENSOR_EXPR_NODE_CONST_SCALAR_H

#include <typeinfo>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: scalar constant container (base class)

    \ingroup libtensor_expr_dag
 **/
class node_const_scalar_base : public node {
public:
    static const char k_op_type[];

public:
    /** \brief Creates the node
     **/
    node_const_scalar_base() :
        node(k_op_type, 0)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_const_scalar_base() { }

    /** \brief Returns the type of scalar
     **/
    virtual const std::type_info &get_type() const = 0;

};


/** \brief Expression node: scalar constant container
    \tparam T Scalar type.

    This node contains a scalar constant.

    \ingroup libtensor_expr_dag
 **/
template<typename T>
class node_const_scalar : public node_const_scalar_base {
private:
    T m_scalar; //!< Scalar

public:
    /** \brief Creates the node
        \param a Reference to scalar.
     **/
    node_const_scalar(const T &a) :
        m_scalar(a)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_const_scalar() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const {
        return new node_const_scalar<T>(*this);
    }

    /** \brief Returns the type of scalar
     **/
    virtual const std::type_info &get_type() const {
        return typeid(T);
    }

    /** \brief Returns the reference to the scalar
     **/
    const T &get_scalar() const {
        return m_scalar;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_CONST_SCALAR_H
