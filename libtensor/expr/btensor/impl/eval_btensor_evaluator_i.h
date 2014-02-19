#ifndef LIBTENSOR_EXPR_EVAL_BTENSOR_EVALUATOR_I_H
#define LIBTENSOR_EXPR_EVAL_BTENSOR_EVALUATOR_I_H

#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/block_tensor/block_tensor_i_traits.h>

namespace libtensor {
namespace expr {


template<size_t N, typename T>
class eval_btensor_evaluator_i {
public:
    typedef block_tensor_i_traits<T> bti_traits;

public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_btensor_evaluator_i() { }

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, bti_traits> &get_bto() const = 0;

};


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_EVAL_BTENSOR_EVALUATOR_I_H
