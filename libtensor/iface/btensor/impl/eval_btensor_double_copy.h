#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H

#include <libtensor/expr/node_ident.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"
#include "interm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class copy {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const expr::node_ident &m_node; //!< Contraction node
    bool m_add; //!< True if add

public:
    copy(const tensor_list &tl, const interm &inter, const expr::node_ident &node, bool add) :
        m_tl(tl), m_interm(inter), m_node(node), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, tid_t tid);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H
