#include <libtensor/block_tensor/bto_symmetrize2.h>
#include <libtensor/block_tensor/bto_symmetrize3.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_transform.h>
#include <libtensor/expr/dag/node_symm.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "eval_btensor_double_autoselect.h"
#include "eval_btensor_double_symm.h"
#include "tensor_from_node.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


template<size_t N, typename T>
class eval_symm_impl : public eval_btensor_evaluator_i<N, T> {
private:
    enum {
        Nmax = symm<N, T>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, T>::bti_traits bti_traits;

private:
    struct dispatch_symm {
        eval_symm_impl &eval;
        const tensor_transf<N, T> &tr;

        dispatch_symm(
            eval_symm_impl &eval_,
            const tensor_transf<N, T> &tr_) :
            eval(eval_), tr(tr_)
        { }

        template<size_t M> void dispatch();
    };

    template<size_t M>
    struct tag { };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    eval_btensor_evaluator_i<N, T> *m_sub; //!< Subexpression
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_symm_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, T> &tr);

    virtual ~eval_symm_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t M>
    void init(const tensor_transf<N, T> &trc, const tag<M>&);

    void init(const tensor_transf<N, T> &trc, const tag<2>&);

    void init(const tensor_transf<N, T> &trc, const tag<3>&);

};


template<size_t N, typename T>
eval_symm_impl<N, T>::eval_symm_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, T> &tr) :

    m_tree(tree), m_id(id), m_sub(0), m_op(0) {

    const node_symm<T> &n =
        m_tree.get_vertex(m_id).template recast_as< node_symm<T> >();

    dispatch_symm disp(*this, tr);
    dispatch_1<2, N>::dispatch(disp, n.get_nsym());
}


template<size_t N, typename T>
eval_symm_impl<N, T>::~eval_symm_impl() {

    delete m_op;
    delete m_sub;
}


template<size_t N, typename T>
void eval_symm_impl<N, T>::init(const tensor_transf<N, T> &tr,
    const tag<2>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_T", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<T> &nn = n.template recast_as< node_symm<T> >();

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
            "libtensor::expr::eval_btensor_T", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }
    size_t nsymidx = nn.get_sym().size() / 2;
    permutation<N> symperm;
    for(size_t i = 0; i < nsymidx; i++) {
        symperm.permute(nn.get_sym()[2*i], nn.get_sym()[2*i+1]);
    }

    tensor_transf<N, T> trinv(tr, true);
    tensor_transf<N, T> trs(symperm, nn.get_pair_tr());
    tensor_transf<N, T> tspr(trinv);
    tspr.transform(trs).transform(tr);

    tensor_transf<N, T> trsub;
    expr_tree::node_id_t rhs = transf_from_node(m_tree, e[0], trsub);
    trsub.transform(tr);
    m_sub = new autoselect<N, T>(m_tree, rhs, trsub);

    m_op = new bto_symmetrize2<N, T>(m_sub->get_bto(), tspr.get_perm(),
        tspr.get_scalar_tr().get_coeff() == 1.0);
}


template<size_t N, typename T>
void eval_symm_impl<N, T>::init(const tensor_transf<N, T> &tr,
    const tag<3>&) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    if(e.size() != 1) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_T", "eval_symm_impl<N>",
            "init()", "Malformed expression (invalid number of children).");
    }

    const node &n = m_tree.get_vertex(m_id);
    const node_symm<T> &nn = n.template recast_as< node_symm<T> >();

    if(nn.get_sym().size() % 3 != 0) {
        throw eval_exception(__FILE__, __LINE__,
            "libtensor::expr::eval_btensor_T", "eval_symm_impl<N>",
            "init()", "Malformed expression (bad symm sequence).");
    }
    size_t nsymidx = nn.get_sym().size() / 3;
    permutation<N> symperm1, symperm2;
    for(size_t i = 0; i < nsymidx; i++) {
        symperm1.permute(nn.get_sym()[3*i], nn.get_sym()[3*i+1]);
        symperm2.permute(nn.get_sym()[3*i], nn.get_sym()[3*i+2]);
    }

    tensor_transf<N, T> trinv(tr, true);
    tensor_transf<N, T> trs1(symperm1, nn.get_pair_tr()),
        trs2(symperm2, nn.get_pair_tr());
    tensor_transf<N, T> tspr1(trinv), tspr2(trinv);
    tspr1.transform(trs1).transform(tr);
    tspr2.transform(trs2).transform(tr);

    tensor_transf<N, T> trsub;
    expr_tree::node_id_t rhs = transf_from_node(m_tree, e[0], trsub);
    trsub.transform(tr);
    m_sub = new autoselect<N, T>(m_tree, rhs, trsub);

    m_op = new bto_symmetrize3<N, T>(m_sub->get_bto(), tspr1.get_perm(),
        tspr2.get_perm(), nn.get_pair_tr().get_coeff() == 1.0);
}


template<size_t N, typename T> template<size_t M>
void eval_symm_impl<N, T>::init(const tensor_transf<N, T> &tr,
    const tag<M>&) {

    throw not_implemented("libtensor::expr::eval_btensor_T",
        "eval_symm_impl<N>", "init()", __FILE__, __LINE__);
}


template<size_t N, typename T> template<size_t M>
void eval_symm_impl<N, T>::dispatch_symm::dispatch() {

    eval.init(tr, tag<M>());
}


} // unnamed namespace


template<size_t N, typename T>
symm<N, T>::symm(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, T> &tr) :

    m_impl(new eval_symm_impl<N, T>(tree, id, tr)) {

}


template<size_t N, typename T>
symm<N, T>::~symm() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_symm {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, T> *tr;
    const node *t;
    symm<N> *e;
    aux_symm() {
#pragma noinline
        { e = new symm<N>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<T>::Nmax,
    aux::aux_symm>;
#endif
template class symm<1, double>;
template class symm<2, double>;
template class symm<3, double>;
template class symm<4, double>;
template class symm<5, double>;
template class symm<6, double>;
template class symm<7, double>;
template class symm<8, double>;

template class symm<1, float>;
template class symm<2, float>;
template class symm<3, float>;
template class symm<4, float>;
template class symm<5, float>;
template class symm<6, float>;
template class symm<7, float>;
template class symm<8, float>;

} // namespace eval_btensor_T
} // namespace expr
} // namespace libtensor
