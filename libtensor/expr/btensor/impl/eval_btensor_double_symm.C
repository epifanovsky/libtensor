#include <libtensor/block_tensor/btod_symmetrize2.h>
#include <libtensor/block_tensor/btod_symmetrize3.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "metaprog.h"
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_symm.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N>
class eval_symm_impl : public eval_btensor_evaluator_i<N, double> {
private:
    enum {
        Nmax = symm<N>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, double>::bti_traits bti_traits;

private:
    struct dispatch_symm {
        eval_symm_impl &eval;
        const tensor_transf<N, double> &tr;
        template<size_t M> void dispatch();
    };

    template<size_t M>
    struct tag { };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    eval_btensor_evaluator_i<N, double> *m_sub; //!< Subexpression
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_symm_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<N, double> &tr);

    virtual ~eval_symm_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t M>
    void init(const tensor_transf<N, double> &trc, const tag<M>&);

    void init(const tensor_transf<N, double> &trc, const tag<2>&);

    void init(const tensor_transf<N, double> &trc, const tag<3>&);

private:
    expr_tree::node_id_t gather_info(const expr_tree &tree,
        expr_tree::node_id_t id, tensor_transf<N, double> &tr);

};


template<size_t N>
eval_symm_impl<N>::eval_symm_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, double> &tr) :

    m_tree(tree), m_id(id), m_sub(0), m_op(0) {

    const node_symm<double> &n =
        m_tree.get_vertex(m_id).recast_as< node_symm<double> >();

    dispatch_symm d = { *this, tr };
    dispatch_1<2, N>::dispatch(d, n.get_nsym());
}


template<size_t N>
eval_symm_impl<N>::~eval_symm_impl() {

    delete m_op;
    delete m_sub;
}


template<size_t N>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<2>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();

    // Need to convert
    // T2 S T1 A -> S' T' A, where S = I + Ts and S' = I + Ts'
    //
    // T2 (I + Ts) T1 A =
    // (T2 T1 + T2 Ts T1) A =
    // (T2 T1 + T2 Ts T2(inv) T2 T1) A =
    // [I + T2 Ts T2(inv)] T2 T1 A
    //
    // => Ts' = T2 Ts T2(inv); T' = T2 T1

    if(nn.get_sym().size() % 2 != 0) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }
    size_t nsymidx = nn.get_sym().size() / 2;
    permutation<N> symperm;
    for(size_t i = 0; i < nsymidx; i++) {
        symperm.permute(nn.get_sym()[2*i], nn.get_sym()[2*i+1]);
    }

    tensor_transf<N, double> tr2(tr), tr2inv(tr2, true);
    tensor_transf<N, double> trs(symperm, nn.get_pair_tr());
    tensor_transf<N, double> tspr(tr2inv);
    tspr.transform(trs);
    tspr.transform(tr2);

    {
        tensor_transf<N, double> trsub;
        expr_tree::node_id_t rhs = gather_info(m_tree, e[0], trsub);
        trsub.transform(tr2);
        m_sub = new autoselect<N>(m_tree, rhs, trsub);
    }

    m_op = new btod_symmetrize2<N>(m_sub->get_bto(), tspr.get_perm(),
        tspr.get_scalar_tr().get_coeff() == 1.0);
}


template<size_t N>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<3>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<double> &nn = n.recast_as< node_symm<double> >();

    if(nn.get_sym().size() != 3) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }

    {
        tensor_transf<N, double> trsub;
        expr_tree::node_id_t rhs = gather_info(m_tree, e[0], trsub);
        trsub.transform(tr);
        m_sub = new autoselect<N>(m_tree, rhs, trsub);
    }

    m_op = new btod_symmetrize3<N>(m_sub->get_bto(), nn.get_sym().at(0),
        nn.get_sym().at(1), nn.get_sym().at(2),
        nn.get_pair_tr().get_coeff() == 1.0);
}


template<size_t N> template<size_t M>
void eval_symm_impl<N>::init(const tensor_transf<N, double> &tr,
    const tag<M>&) {

    throw not_implemented("libtensor::expr::eval_btensor_double",
        "eval_symm_impl<N>", "init()", __FILE__, __LINE__);
}


template<size_t N>
expr_tree::node_id_t eval_symm_impl<N>::gather_info(const expr_tree &tree,
    expr_tree::node_id_t id, tensor_transf<N, double> &tr) {

    const node &n = tree.get_vertex(id);
    if(!n.check_type<node_transform_base>()) return id;

    const node_transform<double> &ntr = n.recast_as< node_transform<double> >();

    const std::vector<size_t> &p = ntr.get_perm();
    if(N != p.size()) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_double",
            "eval_symm_impl<N>", "gather_info()",
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


template<size_t N> template<size_t M>
void eval_symm_impl<N>::dispatch_symm::dispatch() {

    eval.init(tr, tag<M>());
}


} // unnamed namespace


template<size_t N>
symm<N>::symm(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, double> &tr) :

    m_impl(new eval_symm_impl<N>(tree, id, tr)) {

}


template<size_t N>
symm<N>::~symm() {

    delete m_impl;
}


//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_symm {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, double> *tr;
    const node *t;
    symm<N> *e;
    aux_symm() { e = new symm<N>(*tree, id, *tr); }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_symm>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
