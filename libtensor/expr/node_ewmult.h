#ifndef LIBTENSOR_EXPR_NODE_EWMULT_H
#define LIBTENSOR_EXPR_NODE_EWMULT_H

#include <map>
#include "binary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Hadamard tensor product node of the expression tree

    Represents a generalized element-wise multiplication of two expression
    subtrees. The multiplication map determines the indexes of left and right
    arguments which are combined into one. The resulting order of indexes
    are the indexes of the left argument followed by the indexes of the right
    argument which have not been combined by the operation.

    \ingroup libtensor_expr
 **/
class node_ewmult: public binary_node_base {
private:
    std::map<size_t, size_t> m_mmap; //!< Multiplication map

public:
    /** \brief Creates an identity node
        \param left Left argument.
        \param right Right argument.
        \param multmap Multiplication map.
     **/
    node_ewmult(const node &left, const node &right,
            const std::map<size_t, size_t> &multmap) :
        binary_node_base("ewmult", left, right), m_mmap(multmap)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_ewmult() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_ewmult *clone() const {
        return new node_ewmult(*this);
    }

    const std::map<size_t, size_t> &get_mult_map() const {
        return m_mmap;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_EWMULT_H
