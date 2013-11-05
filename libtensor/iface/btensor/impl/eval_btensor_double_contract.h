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

private:
    const tensor_list &m_tl; //!< Tensor list
    const expr::node_contract &m_node; //!< Contraction node
    bool m_add; //!< True if add

public:
    contract(const tensor_list &tl, const expr::node_contract &node, bool add) :
        m_tl(tl), m_node(node), m_add(add)
    { }

    template<size_t NC>
    void evaluate(
        const tensor_transf<NC, double> &trc,
        btensor<NC, double> &btc);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
