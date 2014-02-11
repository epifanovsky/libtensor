#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_DOT_PRODUCT_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_DOT_PRODUCT_H

#include <libtensor/expr/dag/node_dot_product.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class dot_product {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef expr::expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr::expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of diag node

public:
    dot_product(const expr::expr_tree &tr, node_id_t &id) :
        m_tree(tr), m_id(id)
    { }

    void evaluate(node_id_t lhs);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_DOT_PRODUCT_H
