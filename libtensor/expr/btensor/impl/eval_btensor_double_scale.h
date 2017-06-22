#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_SCALE_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_SCALE_H

#include "../eval_btensor.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N, typename T>
class scale {
public:
    enum {
        Nmax = eval_btensor<T>::Nmax
    };

    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree;
    node_id_t m_rhs;

public:
    scale(const expr_tree &tree, node_id_t rhs);
    void evaluate(node_id_t lhs);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_SCALE_H
