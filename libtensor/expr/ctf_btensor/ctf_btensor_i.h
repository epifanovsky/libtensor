#ifndef LIBTENSOR_EXPR_CTF_BTENSOR_I_H
#define LIBTENSOR_EXPR_CTF_BTENSOR_I_H

#include <libtensor/ctf_block_tensor/ctf_block_tensor_i.h>
#include <libtensor/expr/iface/any_tensor_impl.h>
#include "eval_ctf_btensor_holder.h"

namespace libtensor {
namespace expr {


/** \defgroup libtensor_expr_ctf_btensor Expressions with CTF block tensors
    \ingroup libtensor_expr
 **/


/** \brief Distributed CTF block tensor interface
    \tparam N Block tensor order.
    \tparam T Block tensor element type.

    \ingroup libtensor_expr_ctf_btensor
**/
template<size_t N, typename T>
class ctf_btensor_i :
    public any_tensor<N, T>,
    virtual public ctf_block_tensor_rd_i<N, T> {

public:
    static const char k_tensor_type[];

public:
    ctf_btensor_i() : any_tensor<N, T>(*this) {
        eval_ctf_btensor_holder<T>::get_instance().inc_counter();
    }

    virtual ~ctf_btensor_i() {
        eval_ctf_btensor_holder<T>::get_instance().dec_counter();
    }

    virtual const char *get_tensor_type() const {
        return k_tensor_type;
    }

};


template<size_t N, typename T>
const char ctf_btensor_i<N, T>::k_tensor_type[] = "ctf_btensor";


} // namespace expr
} // namespace libtensor


namespace libtensor {

using expr::ctf_btensor_i;

} // namespace libtensor

#endif // LIBTENSOR_EXPR_CTF_BTENSOR_I_H
