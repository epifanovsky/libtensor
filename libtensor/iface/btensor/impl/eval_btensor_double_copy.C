#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/expr/node_ident.h>
#include "metaprog.h"
#include "node_interm.h"
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
    const expr_tree &m_tree; //!< Expression tree
    expr_tree::node_id_t m_id; //!< ID of copy node
    bool m_add; //!< True if add

public:
    eval_copy_impl(const expr_tree &tr, expr_tree::node_id_t id, bool add) :
        m_tree(tr), m_id(id), m_add(add)
    { }

    template<size_t NC>
    void evaluate(const tensor_transf<NC, double> &trc, const node &t);

private:
    template<size_t N>
    btensor_i<N, double> &tensor_from_node(const node &n);

    template<size_t N>
    btensor<N, double> &tensor_from_node(const node &n,
        const block_index_space<N> &bis);

};


template<size_t N>
void eval_copy_impl::evaluate(
    const tensor_transf<N, double> &tr, const node &t) {

    if (N != t.get_n()) {
        throw "Invalid order";
    }

    btensor_i<N, double> &bta = tensor_from_node<N>(m_tree.get_vertex(m_id));
    btod_copy<N> op(bta, tr.get_perm(), tr.get_scalar_tr().get_coeff());
    btensor<N, double> &bt =
            tensor_from_node<N>(m_tree.get_vertex(m_id), op.get_bis());
    if(m_add) {
        op.perform(bt, 1.0);
    } else {
        op.perform(bt);
    }
}


template<size_t N>
btensor_i<N, double> &eval_copy_impl::tensor_from_node(const node &n) {

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
btensor<N, double> &eval_copy_impl::tensor_from_node(const node &n,
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


template<size_t N>
void copy::evaluate(const tensor_transf<N, double> &tr, const node &t) {

    eval_copy_impl(m_tree, m_id, m_add).evaluate(tr, t);
}


//  The code here explicitly instantiates copy::evaluate<N>
namespace copy_ns {
template<size_t N>
struct aux {
    copy *e;
    tensor_transf<N, double> *tr;
    node *n;
    aux() { e->evaluate(*tr, *n); }
};
} // unnamed namespace
template class instantiate_template_1<1, copy::Nmax, copy_ns::aux>;


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor
