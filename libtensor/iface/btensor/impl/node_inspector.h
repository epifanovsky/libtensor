#ifndef LIBTENSOR_NODE_INSPECTOR_H
#define LIBTENSOR_NODE_INSPECTOR_H

#include <libtensor/core/permutation_builder.h>
#include <libtensor/core/tensor_transf_double.h>
#include <libtensor/expr/node.h>
#include <libtensor/expr/node_ident.h>
#include <libtensor/expr/node_transform.h>

namespace libtensor {
namespace iface {
namespace eval_btensor_double {
using namespace libtensor::expr;


template<size_t N>
struct node_with_transf {
    const node &n;
    tensor_transf<N, double> tr;

    node_with_transf(const node &n_) : n(n_) { }
    node_with_transf(const node &n_, const tensor_transf<N, double> &tr_) :
        n(n_), tr(tr_)
    { }

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
        const node_transform<double> &n) const;

};


template<size_t N>
node_with_transf<N> node_inspector::gather_transf() const {

    if(m_node.get_op().compare(expr::node_transform_base::k_op_type) == 0) {

        const node_transform_base &nb = m_node.recast_as<node_transform_base>();
        if(nb.get_type() != typeid(double)) {
            throw "Bad type";
        }
        const node_transform<double> &n =
                nb.recast_as< node_transform<double> >();

        node_with_transf<N> nwt =
            node_inspector(n.get_arg()).template gather_transf<N>();
        nwt.tr.transform(get_tensor_transf<N>(n));
        return nwt;

    }

    return node_with_transf<N>(m_node);
}


template<size_t N>
tensor_transf<N, double> node_inspector::get_tensor_transf(
    const node_transform<double> &n) const {

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


} // namespace eval_btensor_double
} // namespace iface
} // namespace libtensor

#endif // LIBTENSOR_NODE_INSPECTOR_H
