#include <libtensor/block_tensor/btod_contract2.h>
#include <libtensor/block_tensor/btod_scale.h>
#include <libtensor/expr/node_ident.h>
#include "metaprog.h"
#include "node_inspector.h"
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

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    template<size_t NC>
    struct dispatch_contract_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        tid_t tid;
        size_t k, na, nb;
        template<size_t NA> void dispatch();
    };

    template<size_t NC, size_t NA>
    struct dispatch_contract_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        tid_t tid;
        size_t k, na, nb;
        template<size_t K> void dispatch();
    };

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const node_contract &m_node; //!< Contraction node
    bool m_add; //!< True if add

public:
    eval_contract_impl(const tensor_list &tl, const interm &inter,
        const node_contract &node, bool add) :
        m_tl(tl), m_interm(inter), m_node(node), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, tid_t tid);

    template<size_t N, size_t M, size_t K>
    void do_evaluate(const tensor_transf<N + M, double> &trc, tid_t tid);

private:
    template<size_t N>
    btensor<N, double> &tensor_from_tid(tid_t tid);
    template<size_t N>
    btensor<N, double> &tensor_from_tid(tid_t tid,
        const block_index_space<N> &bis);

};


template<size_t NC>
void eval_contract_impl::evaluate(const tensor_transf<NC, double> &trc,
    tid_t tid) {

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

    dispatch_contract_1<NC> d1 = { *this, trc, tid, k, na, nb };
    dispatch_1<1, Nmax>::dispatch(d1, na);
}


template<size_t N, size_t M, size_t K>
void eval_contract_impl::do_evaluate(const tensor_transf<N + M, double> &trc,
    tid_t tid) {

    node_inspector nia(m_node.get_arg(0));
    node_inspector nib(m_node.get_arg(1));
    const node_ident &na = nia.extract_ident();
    const node_ident &nb = nib.extract_ident();

    btensor_i<N + K, double> &bta = tensor_from_tid<N + K>(na.get_tid());
    btensor_i<M + K, double> &btb = tensor_from_tid<M + K>(nb.get_tid());

    contraction2<N, M, K> contr;
    for(typename std::map<size_t, size_t>::const_iterator ic =
            m_node.get_contraction().begin();
            ic != m_node.get_contraction().end(); ++ic) {
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
    btensor<N + M, double> &btc = tensor_from_tid<N + M>(tid, op.get_bis());
    if(m_add) {
        op.perform(btc, trc.get_scalar_tr());
    } else {
        op.perform(btc);
        btod_scale<N + M>(btc, trc.get_scalar_tr()).perform();
    }
}


template<size_t NC> template<size_t NA>
void eval_contract_impl::dispatch_contract_1<NC>::dispatch() {

    enum {
        Kmin = meta_if<(NA > NC), (NA - NC), (NA == NC ? 1 : 0)>::value,
        Kmax = meta_min<NA, (Nmax + NA - NC)/2>::value
    };
    dispatch_contract_2<NC, NA> d2 = { eval, trc, tid, k, na, nb };
    dispatch_1<Kmin, Kmax>::dispatch(d2, k);
}


template<size_t NC, size_t NA> template<size_t K>
void eval_contract_impl::dispatch_contract_2<NC, NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N
    };
    eval.template do_evaluate<N, M, K>(trc, tid);
}


template<size_t N>
btensor<N, double> &eval_contract_impl::tensor_from_tid(tid_t tid) {

    any_tensor<N, double> &anyt = m_tl.get_tensor<N, double>(tid);

    if(m_interm.is_interm(tid)) {
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(anyt);
        if(ph.is_empty()) throw 73;
        return ph.get_btensor();
    } else {
        return btensor<N, double>::from_any_tensor(anyt);
    }
}


template<size_t N>
btensor<N, double> &eval_contract_impl::tensor_from_tid(tid_t tid,
    const block_index_space<N> &bis) {

    any_tensor<N, double> &anyt = m_tl.get_tensor<N, double>(tid);

    if(m_interm.is_interm(tid)) {
        btensor_placeholder<N, double> &ph =
            btensor_placeholder<N, double>::from_any_tensor(anyt);
        if(ph.is_empty()) ph.create_btensor(bis);
        return ph.get_btensor();
    } else {
        return btensor<N, double>::from_any_tensor(anyt);
    }
}


} // unnamed namespace


template<size_t NC>
void contract::evaluate(const tensor_transf<NC, double> &trc, tid_t tid) {

    eval_contract_impl(m_tl, m_interm, m_node, m_add).evaluate(trc, tid);
}


//  The code here explicitly instantiates contract::evaluate<NC>
namespace contract_ns {
template<size_t N>
struct aux {
    contract *e;
    tensor_transf<N, double> *tr;
    aux() { e->evaluate(*tr, 0); }
};
} // unnamed namespace
template class instantiate_template_1<1, contract::Nmax, contract_ns::aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
