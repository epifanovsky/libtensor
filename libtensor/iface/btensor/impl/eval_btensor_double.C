#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/iface/btensor.h>
#include <libtensor/expr/node_contract.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform_double.h>
#include "../eval_btensor.h"

namespace libtensor {
namespace iface {
using namespace libtensor::expr;


void eval_btensor<double>::process_plan(
    const eval_plan &plan, tensor_list &tl) {

    for(eval_plan::iterator i = plan.begin(); i != plan.end(); ++i) {

        const eval_plan_item &item = plan.get_item(i);
        switch(item.code) {
        case eval_plan_action_code::ASSIGN:
            handle_assign(*item.node, tl);
            break;
        case eval_plan_action_code::CREATE_INTERM:
            handle_create_interm(item.tid, tl);
            break;
        case eval_plan_action_code::DELETE_INTERM:
            handle_delete_interm(item.tid, tl);
            break;
        }
    }
}


namespace {

template<size_t Nmin, size_t Nmax>
struct dispatch_1 {
private:
    template<size_t N> struct tag { };

public:
    template<typename Tgt>
    static void dispatch(Tgt &tgt, size_t n) {
        do_dispatch(tag<Nmin>(), tgt, n);
    }

private:
    template<typename Tgt, size_t N>
    static void do_dispatch(const tag<N>&, Tgt &tgt, size_t n) {
        if(N == n) tgt.template dispatch<N>();
        else if(N < n) do_dispatch(tag<N + 1>(), tgt, n);
        else throw "Unable to dispatch";
    }

    template<typename Tgt>
    static void do_dispatch(const tag<Nmax + 1>&, Tgt &tgt, size_t n) {
        throw "Unable to dispatch";
    }

};

template<bool Cond, size_t A, size_t B>
struct meta_if {
    enum {
        value = A
    };
};

template<size_t A, size_t B>
struct meta_if<false, A, B> {
    enum {
        value = B
    };
};

template<size_t A, size_t B>
struct meta_min {
    enum {
        value = meta_if<(A < B), A, B>::value
    };
};

template<size_t A, size_t B>
struct meta_max {
    enum {
        value = meta_if<(A > B), A, B>::value
    };
};

//template<size_t N>
//struct tensor_with_transf {
//    btensor_i<N, double> &bt;
//    tensor_transf<N, double> tr;
//
//    tensor_with_transf(btensor_i<N, double> &bt_) : bt(bt_) { }
//
//};
//
template<size_t N>
struct node_with_transf {
    const node &n;
    tensor_transf<N, double> tr;

    node_with_transf(const node &n_) : n(n_) { }

};

class node_inspector {
private:
    const node &m_node; //!< Expression node

public:
    node_inspector(const node &n) : m_node(n) { }

    template<size_t N>
    node_with_transf<N> gather_transf() const;

    const node_ident &extract_ident() const;

private:
    template<size_t N>
    tensor_transf<N, double> get_tensor_transf(
        const node_transform_double &n) const;

};

template<size_t N>
node_with_transf<N> node_inspector::gather_transf() const {

    if(m_node.get_op().compare("transform") == 0) {

        const node_transform_base &nb = m_node.recast_as<node_transform_base>();
        if(nb.get_type() != typeid(double)) {
            throw "Bad type";
        }
        const node_transform_double &n = nb.recast_as<node_transform_double>();

        node_with_transf<N> nwt =
            node_inspector(n.get_arg()).template gather_transf<N>();
        nwt.tr.transform(get_tensor_transf<N>(n));
        return nwt;

    }

    return node_with_transf<N>(m_node);
}

const node_ident &node_inspector::extract_ident() const {

    if(m_node.get_op().compare("ident") == 0) {

        return m_node.recast_as<node_ident>();

    } else if(m_node.get_op().compare("transform") == 0) {

        const node_transform_base &nb = m_node.recast_as<node_transform_base>();
        return node_inspector(nb.get_arg()).extract_ident();

    }

    throw "No identity node";
}

template<size_t N>
tensor_transf<N, double> node_inspector::get_tensor_transf(
    const node_transform_double &n) const {

    const std::vector<size_t> &p = n.get_perm();
    if(p.size() != N) {
        throw "Bad transform node";
    }
    sequence<N, size_t> s0(0), s1(0);
    for(size_t i = 0; i < N; i++) {
        s0[i] = i;
        s1[i] = p[i];
    }
    permutation_builder<N> pb(s1, s0);
    return tensor_transf<N, double>(pb.get_perm(),
        scalar_transf<double>(n.get_coeff()));
}

class eval_node {
private:
    const tensor_list &m_tl; //!< Tensor list
    const node &m_node; //!< Expression node

public:
    eval_node(const tensor_list &tl, const node &n) :
        m_tl(tl), m_node(n)
    { }

    template<size_t N>
    void evaluate(btensor<N, double> &bt);

};

class eval_contract {
private:
    template<size_t NC>
    struct dispatch_contract_1 {
        eval_contract &eval;
        const tensor_transf<NC, double> &trc;
        btensor<NC, double> &btc;
        size_t k, na, nb;
        template<size_t NA> void dispatch();
    };

    template<size_t NC, size_t NA>
    struct dispatch_contract_2 {
        eval_contract &eval;
        const tensor_transf<NC, double> &trc;
        btensor<NC, double> &btc;
        size_t k, na, nb;
        template<size_t K> void dispatch();
    };

    enum {
        Nmax = eval_btensor<double>::Nmax
    };

private:
    const tensor_list &m_tl; //!< Tensor list
    const node_contract &m_node; //!< Contraction node

public:
    eval_contract(const tensor_list &tl, const node_contract &node) :
        m_tl(tl), m_node(node)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc,
        btensor<NC, double> &btc);

    template<size_t N, size_t M, size_t K>
    void do_evaluate(const tensor_transf<N + M, double> &trc,
        btensor<N + M, double> &btc);

};

template<size_t NC>
void eval_contract::evaluate(const tensor_transf<NC, double> &trc,
    btensor<NC, double> &btc) {

    const node_ident &arga =
        node_inspector(m_node.get_arg(0)).extract_ident();
    const node_ident &argb =
        node_inspector(m_node.get_arg(1)).extract_ident();

    size_t k = m_node.get_contraction().size();
    size_t na = m_tl.get_tensor_order(arga.get_tid());
    size_t nb = m_tl.get_tensor_order(argb.get_tid());

    if(k > na || k > nb) {
        throw "Invalid contraction order";
    }

    dispatch_contract_1<NC> d1 = { *this, trc, btc, k, na, nb };
    dispatch_1<1, NC>::dispatch(d1, na);
}

template<size_t N, size_t M, size_t K>
void eval_contract::do_evaluate(const tensor_transf<N + M, double> &trc,
    btensor<N + M, double> &btc) {

    node_inspector nia(m_node.get_arg(0));
    node_inspector nib(m_node.get_arg(1));
    const node_ident &na = nia.extract_ident();
    const node_ident &nb = nib.extract_ident();

    btensor_i<N + K, double> &bta =
        m_tl.get_tensor<N + K, double>(na.get_tid()).
        template get_tensor< btensor_i<N + K, double> >();
    btensor_i<M + K, double> &btb =
        m_tl.get_tensor<M + K, double>(nb.get_tid()).
        template get_tensor< btensor_i<M + K, double> >();

    contraction2<N, M, K> contr;
    for(typename std::map<size_t, size_t>::const_iterator ic = m_node.get_contraction().begin();
        ic != m_node.get_contraction().end(); ++ic) {
        contr.contract(ic->first, ic->second);
    }

    btod_contract2<N, M, K>(contr, bta, btb).perform(btc);
}

template<size_t NC> template<size_t NA>
void eval_contract::dispatch_contract_1<NC>::dispatch() {

    enum {
        Kmin = meta_if<(NA > NC), NA - NC, 1>::value,
        Kmax = meta_min<NA, (Nmax + NA - NC)/2>::value
    };
    dispatch_contract_2<NC, NA> d2 = { eval, trc, btc, k, na, nb };
    dispatch_1<Kmin, Kmax>::dispatch(d2, k);
}

template<size_t NC, size_t NA> template<size_t K>
void eval_contract::dispatch_contract_2<NC, NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N
    };
    eval.template do_evaluate<N, M, K>(trc, btc);
}

template<size_t N>
void eval_node::evaluate(btensor<N, double> &bt) {

    node_inspector ni(m_node);

    node_with_transf<N> nwt = ni.template gather_transf<N>();

    if(nwt.n.get_op().compare("ident") == 0) {

        const node_ident &n = nwt.n.template recast_as<node_ident>();
        btensor_i<N, double> &bta = m_tl.get_tensor<N, double>(n.get_tid()).
            template get_tensor< btensor_i<N, double> >();
        btod_copy<N>(bta, nwt.tr.get_perm(),
            nwt.tr.get_scalar_tr().get_coeff()).perform(bt);

    } else if(nwt.n.get_op().compare("contract") == 0) {

        const node_contract &n = nwt.n.template recast_as<node_contract>();
        eval_contract(m_tl, n).evaluate(nwt.tr, bt);

    } else {
        throw "Unknown node type";
    }
}

class eval_assign {
private:
    tensor_list &m_tl; //!< Tensor list
    unsigned m_tid; //!< Left-hand-side tensor
    const node &m_rhs; //!< Right-hand side of the assignment

public:
    eval_assign(tensor_list &tl, unsigned tid, const node &rhs) :
        m_tl(tl), m_tid(tid), m_rhs(rhs)
    { }

    template<size_t N>
    void dispatch() {
        btensor<N, double> &bt = btensor<N, double>::from_any_tensor(
            m_tl.get_tensor<N, double>(m_tid));
        eval_node(m_tl, m_rhs).evaluate(bt);
    }

};

} // unnamed namespace


void eval_btensor<double>::handle_assign(
    const expr::node_assign &node, tensor_list &tl) {

    unsigned tid = node.get_tid();
    verify_tensor_type(tid, tl);
    eval_assign e(tl, tid, node.get_rhs());
    dispatch_1<1, Nmax>::dispatch(e, tl.get_tensor_order(tid));
}


void eval_btensor<double>::handle_create_interm(
    unsigned tid, tensor_list &tl) {

}


void eval_btensor<double>::handle_delete_interm(
    unsigned tid, tensor_list &tl) {

}


void eval_btensor<double>::verify_tensor_type(
    unsigned tid, const tensor_list &tl) {

    if(tl.get_tensor_type(tid) != typeid(double)) {
        throw "Bad tensor type";
    }
}


} // namespace iface
} // namespace libtensor
