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

private:
    const tensor_list &m_tl; //!< Tensor list
    const node_ident &m_node; //!< Identity node

public:
    eval_copy_impl(const tensor_list &tl, const node_ident &node) :
        m_tl(tl), m_node(node)
    { }

    template<size_t N>
    void evaluate(
        const tensor_transf<N, double> &tr,
        btensor<N, double> &bt);

};


template<size_t N>
void eval_copy_impl::evaluate(
    const tensor_transf<N, double> &tr,
    btensor<N, double> &bt) {

    if(N != m_tl.get_tensor_order(m_node.get_tid())) {
        throw "Invalid order";
    }

    btensor_i<N, double> &bta = m_tl.get_tensor<N, double>(m_node.get_tid()).
        template get_tensor< btensor_i<N, double> >();
    btod_copy<N>(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff()).
        perform(bt);
}


} // unnamed namespace


template<size_t N>
void copy::evaluate(
    const tensor_transf<N, double> &tr,
    btensor<N, double> &bt) {

    eval_copy_impl(m_tl, m_node).evaluate(tr, bt);
}


//  The code here explicitly instantiates copy::evaluate<N>
namespace {
template<size_t N>
struct aux {
    copy *e;
    tensor_transf<N, double> *tr;
    btensor<N, double> *bt;
    aux() { e->evaluate(*tr, *bt); }
};
} // unnamed namespace
template class instantiate_template_1<1, copy::Nmax, aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
