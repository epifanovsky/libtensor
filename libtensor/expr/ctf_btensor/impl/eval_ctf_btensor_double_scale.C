#include <libtensor/ctf_block_tensor/ctf_btod_scale.h>
#include <libtensor/expr/dag/node_const_scalar.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "ctf_btensor_from_node.h"
#include "eval_ctf_btensor_double_scale.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {


template<size_t N>
scale<N>::scale(const expr_tree &tree, node_id_t rhs) :
    m_tree(tree), m_rhs(rhs) {

}


template<size_t N>
void scale<N>::evaluate(node_id_t lhs) {

    ctf_btensor_from_node<N, double> bt(m_tree, lhs);
    const node_const_scalar<double> &ns = m_tree.get_vertex(m_rhs).
        template recast_as< node_const_scalar<double> >();
    ctf_btod_scale<N>(bt.get_btensor(), ns.get_scalar()).perform();
}


template class scale<1>;
template class scale<2>;
template class scale<3>;
template class scale<4>;
template class scale<5>;
template class scale<6>;
template class scale<7>;
template class scale<8>;


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor
