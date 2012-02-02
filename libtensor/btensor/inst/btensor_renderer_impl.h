#ifndef LIBTENSOR_BTENSOR_RENDERER_IMPL_H
#define LIBTENSOR_BTENSOR_RENDERER_IMPL_H

#include <cstring>
#include <libtensor/btod/btod_scale.h>
#include <libtensor/expr/contract/expression_node_contract2.h>
#include <libtensor/expr/dirprod/expression_node_dirprod.h>
#include <libtensor/expr/dirsum/expression_node_dirsum.h>
#include <libtensor/expr/ident/expression_node_ident.h>
#include "../btensor_i.h"
#include "btensor_renderer_ident.h"
#include "btensor_renderer_contract2_base.h"
#include "btensor_renderer_dirprod_base.h"
#include "btensor_renderer_dirsum_base.h"
#include "btensor_renderer_sum.h"
#include "../btensor_renderer.h"

namespace libtensor {


template<size_t N, typename T>
void btensor_renderer<N, T>::render(const expression<N, T> &e,
    anytensor<N, T> &t) {

    const std::vector< expression_node<N, T>* > &nodes = e.get_nodes();
    if(nodes.empty()) return;

    btensor_i<N, T> &bt = dynamic_cast< btensor_i<N, T>& >(t);

    for(size_t i = 0; i < nodes.size(); i++) {
        std::auto_ptr< btensor_operation_container_i<N, T> > op =
            render_node(*nodes[i]);
        op->perform(i != 0, bt);
    }
}


template<size_t N, typename T>
std::auto_ptr< btensor_operation_container_i<N, T> >
btensor_renderer<N, T>::render(const expression<N, T> &e) {

    std::auto_ptr< btensor_operation_container_i<N, T> > op0;

    const std::vector< expression_node<N, T>* > &nodes = e.get_nodes();

    if(nodes.size() == 0) throw 0;

    if(nodes.size() == 1) {
        return render_node(*nodes[0]);
    }

    std::auto_ptr< btensor_operation_container_sum<N, T> > sum(
        new btensor_operation_container_sum<N, T>());

    for(size_t i = 0; i < nodes.size(); i++) {
        std::auto_ptr< btensor_operation_container_i<N, T> > op =
            render_node(*nodes[i]);
        sum->add_op(op);
    }

    return std::auto_ptr< btensor_operation_container_i<N, T> >(sum);
}


template<size_t N, typename T>
std::auto_ptr< btensor_operation_container_i<N, T> >
btensor_renderer<N, T>::render_node(expression_node<N, T> &n) {

    if(expression_node_ident<N, T>::check_type(n)) {

        return btensor_renderer_ident<N, T>().render_node(
            expression_node_ident<N, T>::cast(n));

    } else if(expression_node_contract2_base<N, T>::check_type(n)) {

        return btensor_renderer_contract2_base<N, T>().render_node(
            expression_node_contract2_base<N, T>::cast(n));

    } else if(expression_node_dirprod_base<N, T>::check_type(n)) {

        return btensor_renderer_dirprod_base<N, T>().render_node(
            expression_node_dirprod_base<N, T>::cast(n));

    } else if(expression_node_dirsum_base<N, T>::check_type(n)) {

        return btensor_renderer_dirsum_base<N, T>().render_node(
            expression_node_dirsum_base<N, T>::cast(n));

    } else {
        throw 0;
    }

    return std::auto_ptr< btensor_operation_container_i<N, T> >();
}


} // namespace libtensor

#endif // LIBTENSOR_BTENSOR_RENDERER_IMPL_H
