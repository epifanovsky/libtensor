#ifndef LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_EVALUATOR_I_H
#define LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_EVALUATOR_I_H

#include <libtensor/gen_block_tensor/additive_gen_bto.h>
#include <libtensor/ctf_block_tensor/ctf_block_tensor_i_traits.h>

namespace libtensor {
namespace expr {


template<size_t N, typename T>
class eval_ctf_btensor_evaluator_i {
public:
    typedef ctf_block_tensor_i_traits<T> bti_traits;

public:
    /** \brief Virtual destructor
     **/
    virtual ~eval_ctf_btensor_evaluator_i() { }

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, bti_traits> &get_bto() const = 0;

};


} // namespace expr
} // namespace libtensor


#endif // LIBTENSOR_EXPR_EVAL_CTF_BTENSOR_EVALUATOR_I_H
