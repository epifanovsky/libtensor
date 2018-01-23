#ifndef LIBTENSOR_EXPR_EVAL_TREE_BUILDER_BTENSOR_H
#define LIBTENSOR_EXPR_EVAL_TREE_BUILDER_BTENSOR_H

#include <libtensor/expr/dag/expr_tree.h>

namespace libtensor {
namespace expr {


template<typename T>
class eval_tree_builder_btensor {
public:
    static const char k_clazz[]; //!< Class name

    typedef std::vector<expr_tree::node_id_t> eval_order_t;

public:
    enum {
        Nmax = 8
    };

private:
    expr_tree m_tree; //!< Evaluation tree
    eval_order_t m_order;

public:
    eval_tree_builder_btensor(const expr_tree &tr) :
        m_tree(tr), m_order(0)
    { }

    /** \brief Modifies the expression tree for direct evaluation
     **/
    void build();

    /** \brief Returns the evaluation tree
     **/
    expr_tree &get_tree() {
        return m_tree;
    }

    const eval_order_t &get_order() {
        return m_order;
    }
};


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_EVAL_TREE_BUILDER_BTENSOR_H
