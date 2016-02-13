#include <libtensor/ctf_block_tensor/ctf_btod_collect.h>
#include <libtensor/ctf_block_tensor/ctf_btod_distribute.h>
#include <libtensor/expr/btensor/btensor.h>
#include <libtensor/expr/ctf_btensor/ctf_btensor.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include <libtensor/expr/btensor/impl/tensor_from_node.h>
#include "ctf_btensor_from_node.h"
#include "eval_ctf_btensor_double_convert.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {
using eval_btensor_double::btensor_from_node;


template<size_t N>
convert<N>::convert(const expr_tree &tree, node_id_t &rhs) :

    m_tree(tree), m_rhs(rhs) {

}


template<size_t N>
void convert<N>::evaluate(node_id_t nid_lhs) {

    const node &rhs = m_tree.get_vertex(m_rhs);
    const node &lhs = m_tree.get_vertex(nid_lhs);

    if(N != lhs.get_n()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_ctf_btensor_double", "convert<N>",
            "evaluate()", "Inconsistent tensor order.");
    }

    const node_ident_any_tensor<N, double> &rhs1 =
        rhs.template recast_as< node_ident_any_tensor<N, double> >();
    const node_ident_any_tensor<N, double> &lhs1 =
        lhs.template recast_as< node_ident_any_tensor<N, double> >();
    std::string ttrhs = rhs1.get_tensor().get_tensor_type();
    std::string ttlhs = lhs1.get_tensor().get_tensor_type();

    if(ttrhs == ctf_btensor_i<N, double>::k_tensor_type &&
            ttlhs == btensor_i<N, double>::k_tensor_type) {

        ctf_btensor_from_node<N, double> dbta(m_tree, m_rhs);
        btensor<N, double> &btb = dynamic_cast< btensor<N, double>& >(
            btensor_from_node<N, double>(m_tree, nid_lhs).get_btensor());
        ctf_btod_collect<N>(dbta.get_btensor()).perform(btb);
    }

    if(ttrhs == btensor_i<N, double>::k_tensor_type &&
            ttlhs == ctf_btensor_i<N, double>::k_tensor_type) {

        btensor_from_node<N, double> bta(m_tree, m_rhs);
        ctf_btensor<N, double> &dbtb = dynamic_cast< ctf_btensor<N, double>& >(
            ctf_btensor_from_node<N, double>(m_tree, nid_lhs).get_btensor());
        ctf_btod_distribute<N>(bta.get_btensor()).perform(dbtb);
    }
}


#if 0
//  The code here explicitly instantiates convert<N>
namespace aux {
template<size_t N>
struct aux_convert {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    expr_tree::node_id_t lhs;
    convert<N> *e;
    aux_convert() {
#pragma noinline
        { e = new convert<N>(*tree, id); e->evaluate(lhs); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_convert>;
#endif
template class convert<1>;
template class convert<2>;
template class convert<3>;
template class convert<4>;
template class convert<5>;
template class convert<6>;
template class convert<7>;
template class convert<8>;


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor
