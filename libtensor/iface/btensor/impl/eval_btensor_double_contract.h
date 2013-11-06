#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H

#include <libtensor/expr/node_contract.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"
#include "interm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class contract {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const expr::node_contract &m_node; //!< Contraction node
    bool m_add; //!< True if add

public:
    contract(const tensor_list &tl, const interm &inter, const expr::node_contract &node, bool add) :
        m_tl(tl), m_interm(inter), m_node(node), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, tid_t tid);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_CONTRACT_H
