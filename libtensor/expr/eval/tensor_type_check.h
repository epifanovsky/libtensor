#ifndef LIBTENSOR_EXPR_TENSOR_TYPE_CHECK_H
#define LIBTENSOR_EXPR_TENSOR_TYPE_CHECK_H

#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/dag/graph.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>

namespace libtensor {
namespace expr {


namespace {

template<typename T, template<size_t NN, typename TT> class Tensor>
struct ttcheck {
private:
    const node_ident &m_n;
    bool m_match;

public:
    ttcheck(const node_ident &n) : m_n(n), m_match(false) { }

    bool is_match() const { return m_match; }

    template<size_t N>
    void dispatch() {
        const node_ident_any_tensor<N, T> &n =
            m_n.recast_as< node_ident_any_tensor<N, T> >();
        const std::string tt(n.get_tensor().get_tensor_type());
        m_match = (tt == Tensor<N, T>::k_tensor_type);
    }

};

} // unnamed namespace


/** \brief Checks that all identity nodes in a graph contain tensors only of
        a given type

    \ingroup libtensor_expr_eval
 **/
template<size_t Nmax, typename T, template<size_t, typename> class Tensor>
bool tensor_type_check(const graph &g) {

    for(graph::iterator i = g.begin(); i != g.end(); ++i) {
        const node &n0 = g.get_vertex(i);
        if(!n0.check_type<node_ident>()) continue;
        const node_ident &n = n0.recast_as<node_ident>();
        if(n.get_type() != typeid(T)) return false;
        ttcheck<T, Tensor> tchk(n);
        eval_btensor_double::dispatch_1<1, Nmax>::dispatch(tchk, n.get_n());
        if(!tchk.is_match()) return false;
    }

    return true;

}


} // namespace expr
} // namespace libtensor

#endif // LIBTENSOR_EXPR_TENSOR_TYPE_CHECK_H

