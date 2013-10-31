#ifndef LIBTENSOR_EXPR_NODE_EWMULT_H
#define LIBTENSOR_EXPR_NODE_EWMULT_H

#include <map>
#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Hadamard tensor product node of the expression tree

    Represents a generalized element-wise multiplication of two expression
    subtrees. Assuming the tensor indexes are arranged successively starting
    with the indexes of the first tensor argument the multiplication map
    determines the index-pairs of the arguments which are combined into one.
    The above order of indexes is retained in the result with only the second
    index of the pairs removed.

    TODO: extend to n tensors

    \ingroup libtensor_expr
 **/
class node_ewmult: public nary_node_base {
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
        nary_node_base("ewmult", 0, left, right), m_mmap(multmap)
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
