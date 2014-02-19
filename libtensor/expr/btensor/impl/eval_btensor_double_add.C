#include <memory>
#include <vector>
#include <libtensor/block_tensor/btod/btod_sum.h>
#include <libtensor/expr/dag/node_add.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h" // for instantiate_template_1
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

private:
    expr_tree::node_id_t gather_info(const expr_tree &tree,
        expr_tree::node_id_t id, tensor_transf<N, double> &tr);

};


template<size_t N>
eval_add_impl<N>::eval_add_impl(const expr_tree &tree,
    expr_tree::node_id_t id) {

    const node_add &n = tree.get_vertex(id).recast_as<node_add>();
    const expr_tree::edge_list_t &e = tree.get_edges_out(id);

    for(size_t i = 0; i < e.size(); i++) {
        tensor_transf<N, double> tr;
        expr_tree::node_id_t rhs = gather_info(tree, e[i], tr);
        m_sub.push_back(new autoselect<N>(tree, rhs, tr, false));
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


template<size_t N>
expr_tree::node_id_t eval_add_impl<N>::gather_info(const expr_tree &tree,
    expr_tree::node_id_t id, tensor_transf<N, double> &tr) {

    const node &n = tree.get_vertex(id);
    if(!n.check_type<node_transform_base>()) return id;

    const node_transform<double> &ntr = n.recast_as< node_transform<double> >();

    const std::vector<size_t> &p = ntr.get_perm();
    if(N != p.size()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double",
            "eval_add_impl<N>", "gather_info()",
            "Malformed expression (bad tensor transformation).");
    }
    sequence<N, size_t> s0(0), s1(0);
    for(size_t i = 0; i < N; i++) {
        s0[i] = i;
        s1[i] = p[i];
    }
    permutation_builder<N> pb(s1, s0);
    tr.permute(pb.get_perm());
    tr.transform(ntr.get_coeff());

    return tree.get_edges_out(id).front();
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
