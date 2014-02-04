#ifndef LIBTENSOR_EXPR_NODE_SCALAR_H
#define LIBTENSOR_EXPR_NODE_SCALAR_H

#include "node_scalar_base.h"

namespace libtensor {
namespace expr {


/** \brief Expression node: scalar container (concrete class)

    \ingroup libtensor_expr
 **/
template<typename T>
class node_scalar : public node_scalar_base {
private:
    T &m_c; //!< Reference to scalar

public:
    /** \brief Creates the node
        \param c Reference to scalar.
     **/
    node_scalar(T &c) : m_c(c)
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

    /** \brief Returns the scalar
     **/
    T &get_c() const {
        return m_c;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_SCALAR_H
