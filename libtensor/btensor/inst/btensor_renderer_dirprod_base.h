#ifndef LIBTENSOR_BTENSOR_RENDERER_DIRPROD_BASE_H
#define LIBTENSOR_BTENSOR_RENDERER_DIRPROD_BASE_H

#include <libtensor/expr/dirprod/expression_node_dirprod.h>
#include <libtensor/btensor/btensor_i.h>
#include "btensor_operation_container_i.h"
#include "btensor_renderer_dirprod.h"
#include "dispatch_size_t.h"

namespace libtensor {


/** \brief Renders expression_node_dirprod_base into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_renderer_dirprod_base {
public:
    std::auto_ptr< btensor_operation_container_i<N, T> > render_node(
        expression_node_dirprod_base<N, T> &n);

};


/** \brief Renders expression_node_dirprod_base into btensor_i
        (specialization for the impossible dirprod case of N = 1)

    \ingroup libtensor_expr
 **/
template<typename T>
class btensor_renderer_dirprod_base<1, T> {
public:
    std::auto_ptr< btensor_operation_container_i<1, T> > render_node(
        expression_node_dirprod_base<1, T> &n) {
        throw 0;
    }

};


template<size_t N, typename T>
struct btensor_renderer_dirprod_dispatch_params {

    expression_node_dirprod_base<N, T> &node;
    std::auto_ptr< btensor_operation_container_i<N, T> > op;

    btensor_renderer_dirprod_dispatch_params(
        expression_node_dirprod_base<N, T> &node_) :
        node(node_)
        { }

};


template<typename T_, size_t NM_, size_t Nmin_, size_t Nmax_>
struct btensor_renderer_dirprod_dispatch_n_traits {

    typedef T_ T;
    enum {
        NM = NM_,
        Nmin = Nmin_,
        Nmax = Nmax_
    };

};


template<size_t N, typename Traits>
struct btensor_renderer_dirprod_dispatch_n {

    enum {
        NM = Traits::NM,
        M = NM - N
    };
    typedef typename Traits::T T;
    typedef btensor_renderer_dirprod_dispatch_params<NM, T> params_t;

    static void dispatch(params_t &par) {

        par.op = btensor_renderer_dirprod<N, M, T>().render_node(
            dynamic_cast< expression_node_dirprod<N, M, T>& >(par.node));
    }

};


template<size_t N, typename T>
std::auto_ptr< btensor_operation_container_i<N, T> >
btensor_renderer_dirprod_base<N, T>::render_node(
    expression_node_dirprod_base<N, T> &n) {

    //  These constants limit the number of contracted indexes
    //  and the maximum number of total indexes.
    //  Required to prevent code bloat.

    enum {
        NM = N, Nmin = 1, Nmax = N - 1
    };
    typedef btensor_renderer_dirprod_dispatch_n_traits<T, NM, Nmin, Nmax>
        traits_t;
    typedef btensor_renderer_dirprod_dispatch_params<NM, T> params_t;
    params_t par(n);
    if(!dispatch_size_t<Nmin, Nmax, btensor_renderer_dirprod_dispatch_n,
        traits_t, params_t>::dispatch(par.node.get_n(), par)) {
        throw 0;
    }

    return par.op;
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_DIRPROD_BASE_H
