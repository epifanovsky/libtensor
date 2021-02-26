#include <libtensor/block_tensor/bto_copy.h>
#ifdef WITH_LIBXM
#include <libtensor/block_tensor/bto_copy_xm.h>
#endif // WITH_LIBXM
#include <libtensor/expr/common/metaprog.h>
#include <libtensor/expr/iface/node_ident_any_tensor.h>
#include <libtensor/expr/eval/eval_exception.h>
#include "tensor_from_node.h"
#include "eval_btensor_double_copy.h"

namespace libtensor {
namespace expr {
namespace eval_btensor_double {


extern bool use_libxm;


namespace {


template<size_t N, typename T>
class eval_copy_impl : public eval_btensor_evaluator_i<N, T> {
private:
    enum {
        Nmax = copy<N, T>::Nmax
    };

public:
    typedef typename eval_btensor_evaluator_i<N, T>::bti_traits bti_traits;

private:
    additive_gen_bto<N, bti_traits> *m_op; //!< Block tensor operation

public:
    eval_copy_impl(const expr_tree &tree, expr_tree::node_id_t id,
        const tensor_transf<N, T> &tr);

    virtual ~eval_copy_impl();

    virtual additive_gen_bto<N, bti_traits> &get_bto() const {
        return *m_op;
    }

};

/*
template<size_t N>
additive_gen_bto<N, typename eval_copy_impl<N, float>::bti_traits> *create_op(
    const expr_tree &tree, expr_tree::node_id_t id,
    const tensor_transf<N, float> &tr) {

    btensor_from_node<N, float> bta(tree, id);


    return new bto_copy<N, float>(bta.get_btensor(), tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
}

template<size_t N>
additive_gen_bto<N, typename eval_copy_impl<N, double>::bti_traits> *create_op(
    const expr_tree &tree, expr_tree::node_id_t id,
    const tensor_transf<N, double> &tr) {

    btensor_from_node<N, double> bta(tree, id);
#ifdef WITH_LIBXM
    if(use_libxm) {
        return new bto_copy_xm<N, double>(bta.get_btensor(), tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    } else {
        return new bto_copy<N, double>(bta.get_btensor(), tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    }
#else // WITH_LIBXM
    return new bto_copy<N, double>(bta.get_btensor(), tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
#endif // WITH_LIBXM
}
*/

template<size_t N, typename T>
additive_gen_bto<N, typename eval_copy_impl<N, T>::bti_traits> *create_op(
    const expr_tree &tree, expr_tree::node_id_t id,
    const tensor_transf<N, T> &tr) {

    btensor_from_node<N, T> bta(tree, id);
#ifdef WITH_LIBXM
    if(use_libxm) {
        return new bto_copy_xm<N, T>(bta.get_btensor(), tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    } else {
        return new bto_copy<N, T>(bta.get_btensor(), tr.get_perm(),
            tr.get_scalar_tr().get_coeff());
    }
#else // WITH_LIBXM
    return new bto_copy<N, T>(bta.get_btensor(), tr.get_perm(),
        tr.get_scalar_tr().get_coeff());
#endif // WITH_LIBXM
}



template<size_t N, typename T>
eval_copy_impl<N, T>::eval_copy_impl(const expr_tree &tree,
    expr_tree::node_id_t id, const tensor_transf<N, T> &tr) {

    m_op = create_op(tree, id, tr);
}


template<size_t N, typename T>
eval_copy_impl<N, T>::~eval_copy_impl() {

    delete m_op;
}


} // unnamed namespace


template<size_t N, typename T>
copy<N, T>::copy(const expr_tree &tree, node_id_t &id,
    const tensor_transf<N, T> &tr) :

    m_impl(new eval_copy_impl<N, T>(tree, id, tr)) {

}


template<size_t N, typename T>
copy<N, T>::~copy() {

    delete m_impl;
}


#if 0
//  The code here explicitly instantiates copy<N>
namespace aux {
template<size_t N>
struct aux_copy {
    const expr_tree *tree;
    expr_tree::node_id_t id;
    const tensor_transf<N, T> *tr;
    const node *t;
    copy<N> *e;
    aux_copy() {
#pragma noinline
        { e = new copy<N>(*tree, id, *tr); }
    }
};
} // namespace aux
template class instantiate_template_1<1, eval_btensor<T>::Nmax,
    aux::aux_copy>;
#endif
template class copy<1, double>;
template class copy<2, double>;
template class copy<3, double>;
template class copy<4, double>;
template class copy<5, double>;
template class copy<6, double>;
template class copy<7, double>;
template class copy<8, double>;

template class copy<1, float>;
template class copy<2, float>;
template class copy<3, float>;
template class copy<4, float>;
template class copy<5, float>;
template class copy<6, float>;
template class copy<7, float>;
template class copy<8, float>;

} // namespace eval_btensor_T
} // namespace expr
} // namespace libtensor
