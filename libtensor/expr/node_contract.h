#ifndef LIBTENSOR_EXPR_NODE_CONTRACT_H
#define LIBTENSOR_EXPR_NODE_CONTRACT_H

#include <map>
#include "nary_node_base.h"

namespace libtensor {
namespace expr {


/** \brief Tensor contraction node of the expression tree

    Represents the contraction of n subexpressions over indexes described
    by the contraction map. Assuming the tensor indexes are arranged
    successively starting with the indexes of the first tensor argument
    the contraction map connects the index pairs of the tensors (key-value
    pairs) over which contractions should be performed.
    The result will then possess a similar index order as above but with the
    contraction index pairs missing. Reordering of result indexes can be
    achieved by a subsequent transformation node.

    For example, the contraction of three tensors
    \f$ \sum_{rs} A_{pr} B_{rs} C_{sq} \f$
    would be represented by the contraction map
    \code { {1,2},{3,4} } \endcode

    \ingroup libtensor_expr
 **/
class node_contract : public nary_node_base {
private:
    std::map<size_t, size_t> m_contr; //!< Contraction map

public:
    /** \brief Creates a contraction node of two tensors
        \param arg1 First argument
        \param arg2 Second argument
        \param contr Contraction map
     **/
    node_contract(const node &arg1, const node &arg2,
        const std::map<size_t, size_t> &contr) :
        nary_node_base("contract", arg1, arg2), m_contr(contr)
    { }

    /** \brief Creates a contraction node of three tensors
        \param arg1 First argument
        \param arg2 Second argument
        \param arg3 Third argument
        \param contr Contraction map
     **/
    node_contract(const node &arg1, const node &arg2, const node &arg3,
        const std::map<size_t, size_t> &contr) :
        nary_node_base("contract", create_args(arg1, arg2, arg3)),
        m_contr(contr)
    { }

    /** \brief Creates a contraction node of n tensors
        \param args List of arguments
        \param contr Contraction map
     **/
    node_contract(
        const std::vector<const node *> &args,
        const std::map<size_t, size_t> &contr) :
        nary_node_base("contract", args), m_contr(contr)
    { }

    /** \brief Virtual destructor
     **/
    virtual ~node_contract() { }

    /** \brief Creates a copy of the node via new
     **/
    virtual node_contract *clone() const {
        return new node_contract(*this);
    }

    const std::map<size_t, size_t> &get_contraction() const {
        return m_contr;
    }

private:
    static std::vector<const node *> create_args(
        const node &n1, const node &n2, const node &n3) {

        std::vector<const node *> args(3);
        args[0] = &n1; args[1] = &n2; args[2] = &n3;
        return args;
    }
};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_NODE_CONTRACT_H
