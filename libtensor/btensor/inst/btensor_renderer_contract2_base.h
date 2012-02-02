#ifndef LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_BASE_H
#define LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_BASE_H

#include <libtensor/expr/contract/expression_node_contract2.h>
#include <libtensor/btensor/btensor_i.h>
#include "btensor_operation_container_i.h"
#include "btensor_renderer_contract2.h"
#include "dispatch_size_t.h"

namespace libtensor {


/** \brief Renders expression_node_contract2_base into btensor_i

    \ingroup libtensor_expr
 **/
template<size_t N, typename T>
class btensor_renderer_contract2_base {
public:
    std::auto_ptr< btensor_operation_container_i<N, T> > render_node(
        expression_node_contract2_base<N, T> &n);

};


template<size_t N, typename T>
struct btensor_renderer_contract2_dispatch_params {

    expression_node_contract2_base<N, T> &node;
    std::auto_ptr< btensor_operation_container_i<N, T> > op;

    btensor_renderer_contract2_dispatch_params(
        expression_node_contract2_base<N, T> &node_) :
        node(node_)
        { }

};


template<typename Tr, size_t N_, size_t M_>
struct btensor_renderer_contract2_dispatch_nm_traits {

    typedef typename Tr::T T;
    enum {
        NM = Tr::NM,
        Kmin = Tr::Kmin,
        Kmax = Tr::Kmax,
        NMmax = Tr::NMmax,
        K = Tr::K,
        N = N_,
        M = M_
    };

};


template<bool Range, typename Traits>
struct btensor_renderer_contract2_dispatch_nm {

    enum {
        NM = Traits::NM
    };
    typedef typename Traits::T T;
    typedef btensor_renderer_contract2_dispatch_params<NM, T> params_t;

    static void dispatch(params_t &par) {

    }

};


template<typename Traits>
struct btensor_renderer_contract2_dispatch_nm<true, Traits> {

    enum {
        NM = Traits::NM,
        K = Traits::K,
        N = Traits::N,
        M = Traits::M
    };
    typedef typename Traits::T T;
    typedef btensor_renderer_contract2_dispatch_params<NM, T> params_t;

    static void dispatch(params_t &par) {

        par.op = btensor_renderer_contract2<N, M, K, T>().render_node(
            dynamic_cast< expression_node_contract2<N, M, K, T>& >(par.node));
    }

};


template<typename Tr, size_t K_>
struct btensor_renderer_contract2_dispatch_n_traits {

    typedef typename Tr::T T;
    enum {
        NM = Tr::NM,
        Kmin = Tr::Kmin,
        Kmax = Tr::Kmax,
        NMmax = Tr::NMmax,
        K = K_
    };

};


template<size_t N, typename Traits>
struct btensor_renderer_contract2_dispatch_n {

    enum {
        NM = Traits::NM,
        NMmax = Traits::NMmax,
        K = Traits::K,
        M = NM - N
    };
    typedef typename Traits::T T;
    typedef btensor_renderer_contract2_dispatch_nm_traits<Traits, N, M>
        traits_t;
    typedef btensor_renderer_contract2_dispatch_params<NM, T> params_t;

    static void dispatch(params_t &par) {
        btensor_renderer_contract2_dispatch_nm<
            (N + K <= NMmax && M + K <= NMmax), traits_t>().dispatch(par);
    }

};


template<typename T_, size_t NM_, size_t Kmin_, size_t Kmax_, size_t NMmax_>
struct btensor_renderer_contract2_dispatch_k_traits {

    typedef T_ T;
    enum {
        NM = NM_,
        Kmin = Kmin_,
        Kmax = Kmax_,
        NMmax = NMmax_
    };

};


template<size_t K, typename Traits>
struct btensor_renderer_contract2_dispatch_k {

    enum {
        NM = Traits::NM,
        Nmin = 0,
        Nmax = NM
    };
    typedef typename Traits::T T;
    typedef btensor_renderer_contract2_dispatch_n_traits<Traits, K> traits_t;
    typedef btensor_renderer_contract2_dispatch_params<NM, T> params_t;

    static void dispatch(params_t &par) {
        if(!dispatch_size_t<Nmin, Nmax, btensor_renderer_contract2_dispatch_n,
            traits_t, params_t>::dispatch(par.node.get_n(), par)) {
            throw 0;
        }
    }

};


template<size_t N, typename T>
std::auto_ptr< btensor_operation_container_i<N, T> >
btensor_renderer_contract2_base<N, T>::render_node(
    expression_node_contract2_base<N, T> &n) {

    //  These constants limit the number of contracted indexes
    //  and the maximum number of total indexes.
    //  Required to prevent code bloat.

    enum {
        NM = N, Kmin = 1, Kmax = 5, NMmax = 6
    };
    typedef btensor_renderer_contract2_dispatch_k_traits<T, NM, Kmin, Kmax,
        NMmax> traits_t;
    typedef btensor_renderer_contract2_dispatch_params<NM, T> params_t;
    params_t par(n);
    if(!dispatch_size_t<Kmin, Kmax, btensor_renderer_contract2_dispatch_k,
        traits_t, params_t>::dispatch(par.node.get_k(), par)) {
        throw 0;
    }

    return par.op;
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_CONTRACT2_BASE_H
