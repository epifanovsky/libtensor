#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_ewmult2.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/expr/dag/node_contract.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include "metaprog.h"
#include "node_interm.h"
#include "eval_btensor_double_contract.h"
#include "tensor_from_node.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {

namespace {


class eval_contract_impl {
public:
    enum {
        Nmax = contract::Nmax
    };

private:
    template<size_t NC>
    struct dispatch_contract_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        const node &t;
        size_t k, na, nb;
        template<size_t NA> void dispatch();
    };

    template<size_t NC, size_t NA>
    struct dispatch_contract_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        const node &t;
        size_t k, na, nb;
        template<size_t K> void dispatch();
    };

    template<size_t NC>
    struct dispatch_ewmult_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        const node &t;
        size_t k, na, nb;
        template<size_t NA> void dispatch();
    };

    template<size_t NC, size_t NA>
    struct dispatch_ewmult_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        const node &t;
        size_t k, na, nb;
        template<size_t K> void dispatch();
    };

private:
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of contraction node
    bool m_add; //!< True if add

public:
    eval_contract_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

    template<size_t N, size_t M, size_t K>
    void do_evaluate_contract(const tensor_transf<N + M, double> &trc,
        const node &t);

    template<size_t N, size_t M, size_t K>
    void do_evaluate_ewmult(const tensor_transf<N + M + K, double> &trc,
        const node &t);

};


template<size_t NC>
void eval_contract_impl::evaluate(const tensor_transf<NC, double> &trc,
    const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_contract &nc = n.recast_as<node_contract>();

    const node &arga = m_tree.get_vertex(e[0]);
    const node &argb = m_tree.get_vertex(e[1]);

    if(nc.do_contract()) {

        size_t k = nc.get_map().size();
        size_t na = arga.get_n();
        size_t nb = argb.get_n();

        if(k > na || k > nb) {
            throw "Invalid contraction order";
        }

        dispatch_contract_1<NC> d1 = { *this, trc, t, k, na, nb };
        dispatch_1<1, Nmax>::dispatch(d1, na);

    } else {

        size_t k = nc.get_map().size();
        size_t na = arga.get_n();
        size_t nb = argb.get_n();

        if(k > na || k > nb) {
            throw "Invalid contraction order";
        }

        dispatch_ewmult_1<NC> d1 = { *this, trc, t, k, na, nb };
        dispatch_1<1, NC>::dispatch(d1, na);

    }
}


template<size_t N, size_t M, size_t K>
void eval_contract_impl::do_evaluate_contract(
    const tensor_transf<N + M, double> &trc, const node &t) {

    const node_contract &n = m_tree.get_vertex(m_id).recast_as<node_contract>();
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

    scalar_transf<double> strc(trc.get_scalar_tr());
    strc.transform(bta.get_transf().get_scalar_tr()).
        transform(btb.get_transf().get_scalar_tr());

    btod_contract2<N, M, K> op(contr, bta.get_btensor(), btb.get_btensor());
    btensor<N + M, double> &btc = tensor_from_node<N + M>(t, op.get_bis());
    if(m_add) {
        op.perform(btc, strc);
    } else {
        op.perform(btc);
        btod_scale<N + M>(btc, strc).perform();
    }
}


template<size_t N, size_t M, size_t K>
void eval_contract_impl::do_evaluate_ewmult(
    const tensor_transf<N + M + K, double> &trc, const node &t) {

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_contract &nc = n.recast_as<node_contract>();

    if(nc.get_map().size() != K) {
        throw 232;
    }

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

    btod_ewmult2<N, M, K> op(bta.get_btensor(), perma, btb.get_btensor(), permb,
        trc1.get_perm(), trc1.get_scalar_tr().get_coeff());
    btensor<N + M + K, double> &btc =
        tensor_from_node<N + M + K>(t, op.get_bis());
    if(m_add) {
        op.perform(btc, 1.0);
    } else {
        op.perform(btc);
    }
}


template<size_t NC> template<size_t NA>
void eval_contract_impl::dispatch_contract_1<NC>::dispatch() {

    enum {
        Kmin = meta_if<(NA > NC), (NA - NC), (NA == NC ? 1 : 0)>::value,
        Kmax = meta_min<NA, (Nmax + NA - NC)/2>::value
    };
    dispatch_contract_2<NC, NA> d2 = { eval, trc, t, k, na, nb };
    dispatch_1<Kmin, Kmax>::dispatch(d2, k);
}


template<size_t NC, size_t NA> template<size_t K>
void eval_contract_impl::dispatch_contract_2<NC, NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N
    };
    eval.template do_evaluate_contract<N, M, K>(trc, t);
}


template<size_t NC> template<size_t NA>
void eval_contract_impl::dispatch_ewmult_1<NC>::dispatch() {

    enum {
        Kmin = 1,
        Kmax = meta_min<NA, NC>::value
    };
    dispatch_ewmult_2<NC, NA> d2 = { eval, trc, t, k, na, nb };
    dispatch_1<Kmin, Kmax>::dispatch(d2, k);
}


template<size_t NC, size_t NA> template<size_t K>
void eval_contract_impl::dispatch_ewmult_2<NC, NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N - K
    };
    eval.template do_evaluate_ewmult<N, M, K>(trc, t);
}


} // unnamed namespace


template<size_t NC>
void contract::evaluate(const tensor_transf<NC, double> &trc, const node &t) {

    eval_contract_impl(m_tree, m_id, m_add).evaluate(trc, t);
}


//  The code here explicitly instantiates contract::evaluate<NC>
namespace aux {
template<size_t N>
struct aux_contract {
    contract *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux_contract() { e->evaluate(*tr, *n); }
};
} // namespace aux
template class instantiate_template_1<1, contract::Nmax, aux::aux_contract>;


} // namespace eval_btensor_double
} // namespace expr
} // namespace libtensor
