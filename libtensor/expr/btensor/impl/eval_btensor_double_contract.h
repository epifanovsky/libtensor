#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_CONTRACT_H

#include "../eval_btensor.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


class contract {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of contraction node
    bool m_add; //!< True if add

public:
    contract(const expr_tree &tr, node_id_t &id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
