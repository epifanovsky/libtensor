#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/expr/node_ident.h>
#include "metaprog.h"
#include "eval_btensor_double_copy.h"

namespace libtensor {
namespace iface {
namespace eval_btensor_double {

namespace {
using namespace libtensor::expr;


class eval_copy_impl {
private:
    enum {
        Nmax = copy::Nmax
    };

    typedef tensor_list::tid_t tid_t; //!< Tensor ID type

private:
    const tensor_list &m_tl; //!< Tensor list
    const interm &m_interm; //!< Intermediates
    const node_ident &m_node; //!< Identity node
    bool m_add; //!< True if add

public:
    eval_copy_impl(const tensor_list &tl, const interm &inter, const node_ident &node, bool add) :
        m_tl(tl), m_interm(inter), m_node(node), m_add(add)
    { }

    template<size_t N>
    void evaluate(const tensor_transf<N, double> &tr, tid_t tid);

private:
    template<size_t N>
    btensor<N, double> &tensor_from_tid(tid_t tid,
        const block_index_space<N> &bis);

};


template<size_t N>
void eval_copy_impl::evaluate(const tensor_transf<N, double> &tr, tid_t tid) {

    if(N != m_tl.get_tensor_order(m_node.get_tid())) {
        throw "Invalid order";
    }

    btensor_i<N, double> &bta = m_tl.get_tensor<N, double>(m_node.get_tid()).
        template get_tensor< btensor_i<N, double> >();
    btod_copy<N> op(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff());
    btensor<N, double> &bt = tensor_from_tid<N>(tid, op.get_bis());
    if(m_add) {
        op.perform(bt, 1.0);
    } else {
        op.perform(bt);
    }
}


template<size_t N>
btensor<N, double> &eval_copy_impl::tensor_from_tid(tid_t tid,
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


template<size_t N>
void copy::evaluate(const tensor_transf<N, double> &tr, tid_t tid) {

    eval_copy_impl(m_tl, m_interm, m_node, m_add).evaluate(tr, tid);
}


//  The code here explicitly instantiates copy::evaluate<N>
namespace {
template<size_t N>
struct aux {
    copy *e;
    tensor_transf<N, double> *tr;
    aux() { e->evaluate(*tr, 0); }
};
} // unnamed namespace
template class instantiate_template_1<1, copy::Nmax, aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
