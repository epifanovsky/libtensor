#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H

#include <libtensor/expr/node_contract.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class contract {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef expr::expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr::expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of contraction node
    bool m_add; //!< True if add

public:
    contract(const expr::expr_tree &tr, node_id_t &id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const expr::node &t);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
