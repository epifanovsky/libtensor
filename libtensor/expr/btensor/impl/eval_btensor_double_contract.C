#include <libtensor/block_tensor/btod_contract2.h>
#ifdef USE_LIBXM
#include <libtensor/block_tensor/btod_contract2_xm.h>
#endif // USE_LIBXM
#include <libtensor/block_tensor/btod_ewmult2.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "tensor_from_node.h"
#include "eval_btensor_double_contract.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


bool use_libxm = false;


namespace {


template<size_t NC>
class eval_contract_impl : public eval_btensor_evaluator_i<NC, double> {
public:
    enum {
        Nmax = contract<NC>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<NC, double>::bti_traits
        bti_traits;

private:
    struct dispatch_contract_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t k, na, nb;

        dispatch_contract_1(
            eval_contract_impl &eval_,
            const tensor_transf<NC, double> &trc_,
            size_t k_, size_t na_, size_t nb_) :
            eval(eval_), trc(trc_), k(k_), na(na_), nb(nb_)
        { }

        template<size_t NA> void dispatch();
    };

    template<size_t NA>
    struct dispatch_contract_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t k, na, nb;

        dispatch_contract_2(
            eval_contract_impl &eval_,
            const tensor_transf<NC, double> &trc_,
            size_t k_, size_t na_, size_t nb_) :
            eval(eval_), trc(trc_), k(k_), na(na_), nb(nb_)
        { }

        template<size_t K> void dispatch();
    };

    struct dispatch_ewmult_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t k, na, nb;

        dispatch_ewmult_1(
            eval_contract_impl &eval_,
            const tensor_transf<NC, double> &trc_,
            size_t k_, size_t na_, size_t nb_) :
            eval(eval_), trc(trc_), k(k_), na(na_), nb(nb_)
        { }

        template<size_t NA> void dispatch();
    };

    template<size_t NA>
    struct dispatch_ewmult_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        size_t k, na, nb;

        dispatch_ewmult_2(
            eval_contract_impl &eval_,
            const tensor_transf<NC, double> &trc_,
            size_t k_, size_t na_, size_t nb_) :
            eval(eval_), trc(trc_), k(k_), na(na_), nb(nb_)
        { }

        template<size_t K> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    additive_gen_bto<NC, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_contract_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<NC, double> &trc);

    virtual ~eval_contract_impl();

    virtual additive_gen_bto<NC, bti_traits> &get_bto() const {
        return *m_op;
    }

    template<size_t N, size_t M, size_t K>
    void init_contract(const tensor_transf<NC, double> &trc);

    template<size_t N, size_t M, size_t K>
    void init_ewmult(const tensor_transf<NC, double> &trc);

};


template<size_t NC>
eval_contract_impl<NC>::eval_contract_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<NC, double> &trc) :

    m_tree(tree), m_id(id), m_op(0) {

    const expr_tree::edge_list_t &e = tree.get_edges_out(id);
    const node_contract &nc = tree.get_vertex(id).recast_as<node_contract>();

    const node &arga = tree.get_vertex(e[0]);
    const node &argb = tree.get_vertex(e[1]);

    if(nc.do_contract()) {

        size_t k = nc.get_map().size();
        size_t na = arga.get_n();
        size_t nb = argb.get_n();

        if(k > na || k > nb) {
            throw eval_exception(__FILE__, __LINE__,
                "libtensor::expr::eval_btensor_double",
                "eval_contract_impl<NC>", "eval_contract_impl()",
                "Invalid contraction order.");
        }

        dispatch_contract_1 disp(*this, trc, k, na, nb);
        dispatch_1<1, Nmax>::dispatch(disp, na);

    } else {

        size_t k = nc.get_map().size();
        size_t na = arga.get_n();
        size_t nb = argb.get_n();

        if(k > na || k > nb) {
            throw eval_exception(__FILE__, __LINE__,
                "libtensor::expr::eval_btensor_double",
                "eval_contract_impl<NC>", "eval_contract_impl()",
                "Invalid product order.");
        }

        dispatch_ewmult_1 disp(*this, trc, k, na, nb);
        dispatch_1<1, NC>::dispatch(disp, na);

    }
}


template<size_t NC>
eval_contract_impl<NC>::~eval_contract_impl() {

    delete m_op;
}


template<size_t NC> template<size_t N, size_t M, size_t K>
void eval_contract_impl<NC>::init_contract(
    const tensor_transf<NC, double> &trc) {

    const node_contract &n =
    		m_tree.get_vertex(m_id).template recast_as<node_contract>();
    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);

    btensor_from_node<N + K, double> bta(m_tree, e[0]);
    btensor_from_node<M + K, double> btb(m_tree, e[1]);

    contraction2<N, M, K> contr;
    for(typename std::multimap<size_t, size_t>::const_iterator ic =
            n.get_map().begin(); ic != n.get_map().end(); ++ic) {

        size_t ka, kb;
        if(ic->first < N + K) {
            ka = ic->first; kb = ic->second - N - K;
        } else {
            ka = ic->second; kb = ic->first - N - K;
        }
        contr.contract(ka, kb);
    }
    contr.permute_a(bta.get_transf().get_perm());
    contr.permute_b(btb.get_transf().get_perm());
    contr.permute_c(trc.get_perm());

#ifdef USE_LIBXM
    if(use_libxm) {
        m_op = new btod_contract2_xm<N, M, K>(contr,
            bta.get_btensor(), bta.get_transf().get_scalar_tr().get_coeff(),
            btb.get_btensor(), btb.get_transf().get_scalar_tr().get_coeff(),
            trc.get_scalar_tr().get_coeff());
    } else {
        m_op = new btod_contract2<N, M, K>(contr,
            bta.get_btensor(), bta.get_transf().get_scalar_tr().get_coeff(),
            btb.get_btensor(), btb.get_transf().get_scalar_tr().get_coeff(),
            trc.get_scalar_tr().get_coeff());
    }
#else // USE_LIBXM
    m_op = new btod_contract2<N, M, K>(contr,
        bta.get_btensor(), bta.get_transf().get_scalar_tr().get_coeff(),
        btb.get_btensor(), btb.get_transf().get_scalar_tr().get_coeff(),
        trc.get_scalar_tr().get_coeff());
#endif // USE_LIBXM
}


template<size_t NC> template<size_t N, size_t M, size_t K>
void eval_contract_impl<NC>::init_ewmult(const tensor_transf<NC, double> &trc) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node_contract &nc = m_tree.get_vertex(m_id).
        template recast_as<node_contract>();

    btensor_from_node<N + K, double> bta(m_tree, e[0]);
    btensor_from_node<M + K, double> btb(m_tree, e[1]);

    sequence<N + K, size_t> seqa1, seqa2;
    sequence<M + K, size_t> seqb1, seqb2;
    sequence<N + M + K, size_t> seqc1, seqc2;
    mask<N + K> ma;
    mask<M + K> mb;
    for(size_t i = 0; i < N + K; i++) seqa1[i] = i;
    for(size_t i = 0; i < M + K; i++) seqb1[i] = i;

    size_t j = 0;
    for(typename std::multimap<size_t, size_t>::const_iterator ic =
            nc.get_map().begin(); ic != nc.get_map().end(); ++ic, j++) {
        seqa2[N + j] = ic->first;
        ma[ic->first] = true;
        seqb2[M + j] = ic->second;
        mb[ic->second] = true;
        seqc2[N + M + j] = ic->first;
    }
    for(size_t i = 0, j = 0; i < N + K; i++) if(!ma[i]) {
        seqa2[j] = i;
        seqc2[j] = i;
        j++;
    }
    for(size_t i = 0, j = 0; i < M + K; i++) if(!mb[i]) {
        seqb2[j] = i;
        seqc2[N + j] = N + K + i;
        j++;
    }
    for(size_t i = 0; i < N + K; i++) seqc1[i] = i;
    for(size_t i = 0, j = 0; i < M + K; i++) if(!mb[i]) {
        seqc1[N + K + j] = N + K + i;
        j++;
    }

    permutation_builder<N + K> pba(seqa2, seqa1);
    permutation_builder<M + K> pbb(seqb2, seqb1);
    permutation_builder<N + M + K> pbc(seqc1, seqc2);
    permutation<N + K> perma(bta.get_transf().get_perm());
    perma.permute(pba.get_perm());
    permutation<M + K> permb(btb.get_transf().get_perm());
    permb.permute(pbb.get_perm());

    tensor_transf<N + M + K, double> trc1(pbc.get_perm());
    trc1.transform(trc);
    trc1.transform(bta.get_transf().get_scalar_tr());
    trc1.transform(btb.get_transf().get_scalar_tr());

    m_op = new btod_ewmult2<N, M, K>(bta.get_btensor(), perma,
        btb.get_btensor(), permb, trc1.get_perm(),
        trc1.get_scalar_tr().get_coeff());
}


template<size_t NC> template<size_t NA>
void eval_contract_impl<NC>::dispatch_contract_1::dispatch() {

    enum {
        Kmin = meta_if<(NA > NC), (NA - NC), (NA == NC ? 1 : 0)>::value,
        Kmax = meta_min<NA, (Nmax + NA - NC)/2>::value
    };
    dispatch_contract_2<NA> disp(eval, trc, k, na, nb);
    dispatch_1<Kmin, Kmax>::dispatch(disp, k);
}


template<size_t NC> template<size_t NA> template<size_t K>
void eval_contract_impl<NC>::dispatch_contract_2<NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N
    };
    eval.template init_contract<N, M, K>(trc);
}


template<size_t NC> template<size_t NA>
void eval_contract_impl<NC>::dispatch_ewmult_1::dispatch() {

    enum {
        Kmin = 1,
        Kmax = meta_min<NA, NC>::value
    };
    dispatch_ewmult_2<NA> disp(eval, trc, k, na, nb);
    dispatch_1<Kmin, Kmax>::dispatch(disp, k);
}


template<size_t NC> template<size_t NA> template<size_t K>
void eval_contract_impl<NC>::dispatch_ewmult_2<NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N - K
    };
    eval.template init_ewmult<N, M, K>(trc);
}


} // unnamed namespace


template<size_t NC>
contract<NC>::contract(const expr_tree &tree, node_id_t &id,
    const tensor_transf<NC, double> &tr) :

    m_impl(new eval_contract_impl<NC>(tree, id, tr)) {

}


template<size_t NC>
contract<NC>::~contract() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates contract<NC>
namespace aux {
template<size_t NC>
struct aux_contract {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<NC, double> *tr;
    const node *t;
    contract<NC> *e;
    aux_contract() {
#pragma noinline
        { e = new contract<NC>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<double>::Nmax,
    aux::aux_contract>;
#endif
template class contract<1>;
template class contract<2>;
template class contract<3>;
template class contract<4>;
template class contract<5>;
template class contract<6>;
template class contract<7>;
template class contract<8>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
