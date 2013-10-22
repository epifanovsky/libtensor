#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H

#include <libtensor/expr/node_ident.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class copy {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

private:
    const tensor_list &m_tl; //!< Tensor list
    const expr::node_ident &m_node; //!< Contraction node

public:
    copy(const tensor_list &tl, const expr::node_ident &node) :
        m_tl(tl), m_node(node)
    { }

    template<size_t N>
    void evaluate(
        const tensor_transf<N, double> &tr,
        btensor<N, double> &bt);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_COPY_H
