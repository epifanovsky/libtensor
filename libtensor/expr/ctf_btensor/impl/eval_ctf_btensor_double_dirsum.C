#include <libtensor/core/permutation_builder.h>
#include <libtensor/ctf_block_tensor/ctf_btod_dirsum.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_dirsum.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "ctf_btensor_from_node.h"
#include "eval_ctf_btensor_double_dirsum.h"

namespace libtensor {
namespace expr {
namespace eval_ctf_btensor_double {

namespace {
using eval_btensor_double::dispatch_1;


template<size_t NC>
class eval_dirsum_impl : public eval_ctf_btensor_evaluator_i<NC, double> {
public:
    enum {
        Nmax = dirsum<NC>::Nmax
    };

public:
    typedef typename eval_ctf_btensor_evaluator_i<NC, double>::bti_traits
        bti_traits;

private:
    struct dispatch_dirsum {
        eval_dirsum_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t na, nb;

        dispatch_dirsum(
            eval_dirsum_impl &eval_,
            const tensor_transf<NC, double> &trc_,
            size_t na_, size_t nb_) :
            eval(eval_), trc(trc_), na(na_), nb(nb_)
        { }

        template<size_t NA> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    additive_gen_bto<NC, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_dirsum_impl(const expr_tree &tr, expr_tree::node_id_t id,
        const tensor_transf<NC, double> &trc);

    virtual ~eval_dirsum_impl();

    virtual additive_gen_bto<NC, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t NA, size_t NB>
    void init(const tensor_transf<NC, double> &trc);

};


template<size_t NC>
eval_dirsum_impl<NC>::eval_dirsum_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<NC, double> &trc) :

    m_tree(tree), m_id(id), m_op(0) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_dirsum &nd = n.recast_as<node_dirsum>();

    const node &arga = m_tree.get_vertex(e[0]);
    const node &argb = m_tree.get_vertex(e[1]);

    size_t na = arga.get_n();
    size_t nb = argb.get_n();

    dispatch_dirsum disp(*this, trc, na, nb);
    dispatch_1<1, NC - 1>::dispatch(disp, na);
}


template<size_t NC>
eval_dirsum_impl<NC>::~eval_dirsum_impl() {

    delete m_op;
}


template<size_t NC> template<size_t NA, size_t NB>
void eval_dirsum_impl<NC>::init(const tensor_transf<NC, double> &trc) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node_dirsum &nd =
    		m_tree.get_vertex(m_id).template recast_as<node_dirsum>();

    ctf_btensor_from_node<NA, double> bta(m_tree, e[0]);
    ctf_btensor_from_node<NB, double> btb(m_tree, e[1]);

    m_op = new ctf_btod_dirsum<NA, NB>(bta.get_btensor(),
        bta.get_transf().get_scalar_tr(), btb.get_btensor(),
        btb.get_transf().get_scalar_tr(), trc);
}


template<size_t NC> template<size_t NA>
void eval_dirsum_impl<NC>::dispatch_dirsum::dispatch() {

    enum {
        NB = NC - NA
    };
    eval.template init<NA, NB>(trc);
}


} // unnamed namespace


template<size_t NC>
dirsum<NC>::dirsum(const expr_tree &tree, node_id_t &id,
    const tensor_transf<NC, double> &tr) :

    m_impl(new eval_dirsum_impl<NC>(tree, id, tr)) {

}


template<size_t NC>
dirsum<NC>::~dirsum() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t NC>
struct aux_dirsum {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<NC, double> *tr;
    const node *t;
    dirsum<NC> *e;
    aux_dirsum() {
#pragma noinline
        { e = new dirsum<NC>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_dirsum>;
#endif
template class dirsum<1>;
template class dirsum<2>;
template class dirsum<3>;
template class dirsum<4>;
template class dirsum<5>;
template class dirsum<6>;
template class dirsum<7>;
template class dirsum<8>;


} // namespace eval_ctf_btensor_double
} // namespace expr
} // namespace libtensor
