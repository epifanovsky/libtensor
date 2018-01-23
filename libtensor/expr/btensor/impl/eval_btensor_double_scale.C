#include <libtensor/block_tensor/bto_scale.h>
#include <libtensor/expr/dag/node_const_scalar.h>
#include <libtensor/expr/dag/node_set.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "tensor_from_node.h"
#include "eval_btensor_double_scale.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


template<size_t N, typename T>
scale<N, T>::scale(const expr_tree &tree, node_id_t rhs) :
    m_tree(tree), m_rhs(rhs) {

}


template<size_t N, typename T>
void scale<N, T>::evaluate(node_id_t lhs) {

    btensor_from_node<N, T> bt(m_tree, lhs);
    const node_const_scalar<T> &ns = m_tree.get_vertex(m_rhs).
        template recast_as< node_const_scalar<T> >();
    bto_scale<N, T>(bt.get_or_create_btensor(bt.get_btensor().get_bis()),
        ns.get_scalar()).perform();
}


template class scale<1, double>;
template class scale<2, double>;
template class scale<3, double>;
template class scale<4, double>;
template class scale<5, double>;
template class scale<6, double>;
template class scale<7, double>;
template class scale<8, double>;

template class scale<1, float>;
template class scale<2, float>;
template class scale<3, float>;
template class scale<4, float>;
template class scale<5, float>;
template class scale<6, float>;
template class scale<7, float>;
template class scale<8, float>;

} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
