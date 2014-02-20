#include <memory>
#include <vector>
#include <libtensor/block_tensor/btod/btod_sum.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h" // for instantiate_template_1
#include "tensor_from_node.h"
#include "eval_btensor_double_add.h"
#include "eval_btensor_double_autoselect.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {
using std::auto_ptr;


template<size_t N>
class eval_add_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = add<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    std::vector<eval_btensor_evaluator_i<N, double>*> m_sub; //!< Subexpressions
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_add_impl(const expr_tree &tr, expr_tree::node_id_t id);

    virtual ~eval_add_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};


template<size_t N>
eval_add_impl<N>::eval_add_impl(const expr_tree &tree,
    expr_tree::node_id_t id) {

    const node_add &n = tree.get_vertex(id).recast_as<node_add>();
    const expr_tree::edge_list_t &e = tree.get_edges_out(id);

    for(size_t i = 0; i < e.size(); i++) {
        tensor_transf<N, double> tr;
        expr_tree::node_id_t rhs = transf_from_node(tree, e[i], tr);
        m_sub.push_back(new autoselect<N>(tree, rhs, tr));
    }

    auto_ptr< btod_sum<N> > op;
    for(size_t i = 0; i < m_sub.size(); i++) {
        if(i == 0) {
            op.reset(new btod_sum<N>(m_sub[0]->get_bto()));
        } else {
            op->add_op(m_sub[i]->get_bto());
        }
    }
    m_op = op.release();
}


template<size_t N>
eval_add_impl<N>::~eval_add_impl() {

    delete m_op;
    for(size_t i = 0; i < m_sub.size(); i++) delete m_sub[i];
}


} // unnamed namespace


template<size_t N>
add<N>::add(const expr_tree &tree, node_id_t id) :

    m_impl(new eval_add_impl<N>(tree, id)) {

}


template<size_t N>
add<N>::~add() {

    delete m_impl;
}


//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_add {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    add<N> *e;
    aux_add() { e = new add<N>(*tree, id); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_add>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
