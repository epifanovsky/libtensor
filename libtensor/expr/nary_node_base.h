#ifndef LIBTENSOR_EXPR_NARY_NODE_BASE_H
#define LIBTENSOR_EXPR_NARY_NODE_BASE_H

#include <vector>
#include "node.h"

namespace libtensor {
namespace expr {


/** \brief Base class of n-ary tensor operation node of the expression tree

    Base class of tensor operations with more than one argument

    \ingroup libtensor_expr
 **/
class nary_node_base : public node {
private:
    std::vector<const node *> m_args; //!< Arguments

public:
    /** \brief Creates a binary node
        \param op Operation name
        \param arg1 First argument
        \param arg2 Second argument
     **/
    nary_node_base(const std::string &op, const node &arg1,
        const node &arg2) : node(op), m_args(2) {

        m_args[0] = &arg1; m_args[1] = &arg2;
    }

    /** \brief Creates a n-ary node
        \param op Operation name
        \param args List of arguments
     **/
    nary_node_base(const std::string &op,
        const std::vector<const node *> &args) : node(op), m_args(args)
    { }


    /** \brief Virtual destructor
     **/
    virtual ~nary_node_base() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node *clone() const = 0;

    /** \brief Return number of arguments
     **/
    size_t get_nargs() const {
        return m_args.size();
    }

    /** \brief Returns the i-th argument
     **/
    const node &get_arg(size_t i) const;
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_BINARY_NODE_BASE_H
