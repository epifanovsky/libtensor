#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H

#include "../eval_btensor.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


class dirsum {
public:
    enum {
        Nmax = eval_btensor<double>::Nmax
    };

    typedef expr_tree::node_id_t node_id_t; //!< Node ID type

private:
    const expr_tree &m_tree; //!< Expression tree
    node_id_t m_id; //!< ID of diag node
    bool m_add; //!< True if add

public:
    dirsum(const expr_tree &tr, node_id_t &id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, const node &t);

};


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_DOUBLE_DIRSUM_H
