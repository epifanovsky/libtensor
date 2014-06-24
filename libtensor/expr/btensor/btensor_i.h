#ifndef LIBTENSOR_EXPR_BTENSOR_I_H
#define LIBTENSOR_EXPR_BTENSOR_I_H

#include <libtensor/block_tensor/block_tensor_i.h>
#include <libtensor/expr/iface/any_tensor_impl.h>
#include "eval_btensor_holder.h"

namespace libtensor {
namespace expr {


/** \defgroup libtensor_expr_btensor Expressions with block tensors
    \ingroup libtensor_expr
 **/


/** \brief Block tensor interface
    \tparam N Block tensor order.
    \tparam T Block tensor element type.

    \ingroup libtensor_expr_btensor
 **/
template<size_t N, typename T>
class btensor_i :
    public any_tensor<N, T>,
    virtual public block_tensor_rd_i<N, T> {

public:
    static const char k_tensor_type[];

public:
    btensor_i() : any_tensor<N, T>(*this) {
        eval_btensor_holder<T>::get_instance().inc_counter();
    }

    virtual ~btensor_i() {
        eval_btensor_holder<T>::get_instance().dec_counter();
    }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

};


template<size_t N, typename T>
const char btensor_i<N, T>::k_tensor_type[] = "btensor";


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::btensor_i;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_BTENSOR_I_H
