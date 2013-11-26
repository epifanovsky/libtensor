#ifndef LIBTENSOR_IFACE_EVAL_TREE_BUILDER_BTENSOR_H
#define LIBTENSOR_IFACE_EVAL_TREE_BUILDER_BTENSOR_H

#include <libtensor/expr/expr_tree.h>

namespace libtensor {
namespace iface {


class eval_tree_builder_btensor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        Nmax = 8
    };

private:
    expr::expr_tree m_tree; //!< Expression tree

public:
    eval_tree_builder_btensor(const expr::expr_tree &tr) :

        m_tree(tr)
    { }

    /** \brief Modifies the expression tree for direct evaluation
     **/
    void build();

    /** \brief Returns the evaluation plan
     **/
    expr::expr_tree &get_tree() {
        return m_tree;
    }

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_TREE_BUILDER_BTENSOR_H
