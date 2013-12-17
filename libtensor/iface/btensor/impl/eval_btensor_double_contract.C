#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_ewmult2.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_ident.h>
#include "metaprog.h"
#include "node_interm.h"
#include "eval_btensor_double_contract.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


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

private:
    template<size_t N>
    btensor_i<N, double> &tensor_from_node(const node &n);
    template<size_t N>
    btensor<N, double> &tensor_from_node(const node &n,
        const block_index_space<N> &bis);

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

    const expr_tree::edge_list_t &e = m_tree.get_edges_out(m_id);
    const node &n = m_tree.get_vertex(m_id);
    const node_contract &nc = n.recast_as<node_contract>();

    btensor_i<N + K, double> &bta =
            tensor_from_node<N + K>(m_tree.get_vertex(e[0]));
    btensor_i<M + K, double> &btb =
            tensor_from_node<M + K>(m_tree.get_vertex(e[1]));

    contraction2<N, M, K> contr;
    for(typename std::multimap<size_t, size_t>::const_iterator ic =
            nc.get_map().begin(); ic != nc.get_map().end(); ++ic) {

        size_t ka, kb;
        if(ic->first < N + K) {
            ka = ic->first; kb = ic->second - N - K;
        } else {
            ka = ic->second; kb = ic->first - N - K;
        }
        contr.contract(ka, kb);
    }
    contr.permute_c(trc.get_perm());

    btod_contract2<N, M, K> op(contr, bta, btb);
    btensor<N + M, double> &btc = tensor_from_node<N + M>(t, op.get_bis());
    if(m_add) {
        op.perform(btc, trc.get_scalar_tr());
    } else {
        op.perform(btc);
        btod_scale<N + M>(btc, trc.get_scalar_tr()).perform();
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

    btensor_i<N + K, double> &bta =
            tensor_from_node<N + K>(m_tree.get_vertex(e[0]));
    btensor_i<M + K, double> &btb =
            tensor_from_node<M + K>(m_tree.get_vertex(e[1]));

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

    tensor_transf<N + M + K, double> trc1(pbc.get_perm());
    trc1.transform(trc);

    btod_ewmult2<N, M, K> op(bta, pba.get_perm(), btb, pbb.get_perm(),
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


template<size_t N>
btensor_i<N, double> &eval_contract_impl::tensor_from_node(const node &n) {

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

        return ni.get_tensor().template get_tensor< btensor_i<N, double> >();
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) throw 73;
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


template<size_t N>
btensor<N, double> &eval_contract_impl::tensor_from_node(const node &n,
    const block_index_space<N> &bis) {

    if (n.get_op().compare(node_ident_base::k_op_type) == 0) {
        const node_ident<N, double> &ni =
                n.recast_as< node_ident<N, double> >();

        return btensor<N, double>::from_any_tensor(ni.get_tensor());
    }
    else if (n.get_op().compare(node_interm_base::k_op_type) == 0) {

        const node_interm<N, double> &ni =
                n.recast_as< node_interm<N, double> >();
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(ni.get_tensor());

        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();
    }
    else {
        throw 74;
    }
}


} // unnamed namespace


template<size_t NC>
void contract::evaluate(const tensor_transf<NC, double> &trc, const node &t) {

    eval_contract_impl(m_tree, m_id, m_add).evaluate(trc, t);
}


//  The code here explicitly instantiates contract::evaluate<NC>
namespace contract_ns {
template<size_t N>
struct aux {
    contract *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux() { e->evaluate(*tr, *n); }
};
} // unnamed namespace
template class instantiate_template_1<1, contract::Nmax, contract_ns::aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
