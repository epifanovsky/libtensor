#ifndef LIBTENSOR_EXPR_EVAL_PLAN_H
#define LIBTENSOR_EXPR_EVAL_PLAN_H

#include <vector>
#include "node_assign.h"

namespace libtensor {
namespace expr {


/** \brief Evaluation plan of an expression

    Contains a list of instructions for evaluating an expression. Each node
    represents one assignment instruction.

    \sa node_assign

    \ingroup libtensor_expr
 **/
class eval_plan {
private:
    std::vector<node_assign*> m_lst; //!< List of nodes

public:
    typedef std::vector<node_assign*>::const_iterator iterator; //!< Iterator

public:
    /** \brief Destroys the object
     **/
    ~eval_plan() {
        for(size_t i = 0; i < m_lst.size(); i++) delete m_lst[i];
    }

    /** \brief Returns the iterator to the first node
     **/
    iterator begin() const {
        return m_lst.begin();
    }

    /** \brief Returns the iterator to after the last node
     **/
    iterator end() const {
        return m_lst.end();
    }

    /** \brief Appends a node to the end of the list
     **/
    void add_node(const node_assign &n) {
        m_lst.push_back(new node_assign(n));
    }

    /** \brief Returns the node pointed by the given iterator
     **/
    const node_assign &get_node(const iterator &i) const {
        return **i;
    }

};


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_PLAN_H
