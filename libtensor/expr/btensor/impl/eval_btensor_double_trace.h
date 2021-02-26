#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_TRACE_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_TRACE_H

#include "../eval_btensor.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

template<typename T>
class trace {
public:
    enum {
        Nmax = eval_btensor<T>::Nmax
    };

    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of diag node

public:
    trace(const expr_tree &tr, node_id_t &id) :
        m_tree(tr), m_id(id)
    { }

    void evaluate(node_id_t lhs);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_TRACE_H
