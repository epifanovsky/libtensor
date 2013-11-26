#ifndef LIBTENSOR_EVAL_BTENSOR_DOUBLE_SYMM_H
#define LIBTENSOR_EVAL_BTENSOR_DOUBLE_SYMM_H

#include <libtensor/expr/node_symm.h>
#include <libtensor/iface/btensor.h>
#include "../eval_btensor.h"
#include "interm.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {


class symm {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef expr::expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr::expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of symmetrization node
    bool m_add; //!< True if add

public:
    symm(const expr::expr_tree &tr, node_id_t &id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, const expr::node &t);

};


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_EVAL_BTENSOR_DOUBLE_SYMM_H
