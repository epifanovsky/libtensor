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
private:
    template<size_t NC>
    struct dispatch_contract_1 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        btensor<NC, double> &btc;
        size_t k, na, nb;
        template<size_t NA> void dispatch();
    };

    template<size_t NC, size_t NA>
    struct dispatch_contract_2 {
        eval_contract_impl &eval;
        const tensor_transf<NC, double> &trc;
        btensor<NC, double> &btc;
        size_t k, na, nb;
        template<size_t K> void dispatch();
    };

    enum {
        Nmax = contract::Nmax
    };

private:
    const tensor_list &m_tl; //!< Tensor list
    const node_contract &m_node; //!< Contraction node
    bool m_add; //!< True if add

public:
    eval_contract_impl(const tensor_list &tl, const node_contract &node,
        bool add) :
        m_tl(tl), m_node(node), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc,
        btensor<NC, double> &btc);

    template<size_t N, size_t M, size_t K>
    void do_evaluate(const tensor_transf<N + M, double> &trc,
        btensor<N + M, double> &btc);

};


template<size_t NC>
void eval_contract_impl::evaluate(const tensor_transf<NC, double> &trc,
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
    dispatch_1<1, Nmax>::dispatch(d1, na);
}


template<size_t N, size_t M, size_t K>
void eval_contract_impl::do_evaluate(const tensor_transf<N + M, double> &trc,
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

    if(m_add) {
        btod_contract2<N, M, K>(contr, bta, btb).
            perform(btc, trc.get_scalar_tr());
    } else {
        btod_contract2<N, M, K>(contr, bta, btb).perform(btc);
        btod_scale<N + M>(btc, trc.get_scalar_tr()).perform();
    }
}


template<size_t NC> template<size_t NA>
void eval_contract_impl::dispatch_contract_1<NC>::dispatch() {

    enum {
        Kmin = meta_if<(NA > NC), NA - NC, 1>::value,
        Kmax = meta_min<NA, (Nmax + NA - NC)/2>::value
    };
    dispatch_contract_2<NC, NA> d2 = { eval, trc, btc, k, na, nb };
    dispatch_1<Kmin, Kmax>::dispatch(d2, k);
}


template<size_t NC, size_t NA> template<size_t K>
void eval_contract_impl::dispatch_contract_2<NC, NA>::dispatch() {

    enum {
        N = NA - K,
        M = NC - N
    };
    eval.template do_evaluate<N, M, K>(trc, btc);
}


} // unnamed namespace


template<size_t NC>
void contract::evaluate(
    const tensor_transf<NC, double> &trc,
    btensor<NC, double> &btc) {

    eval_contract_impl(m_tl, m_node, m_add).evaluate(trc, btc);
}


//  The code here explicitly instantiates contract::evaluate<NC>
namespace {
template<size_t N>
struct aux {
    contract *e;
    tensor_transf<N, double> *tr;
    btensor<N, double> *bt;
    aux() { e->evaluate(*tr, *bt); }
};
} // unnamed namespace
template class instantiate_template_1<1, contract::Nmax, aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
