#ifndef LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H
#define LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H

#include <libtensor/block_tensor/btod_traits.h>
#include <libtensor/block_tensor/btod_add.h>
#include <libtensor/block_tensor/btod_copy.h>
#include <libtensor/block_tensor/btod_set.h>
#include <libtensor/block_tensor/btod/btod_sum.h>
#include <libtensor/gen_block_tensor/additive_gen_bto.h>

namespace libtensor {
namespace labeled_btensor_expr {


template<size_t N>
class evalfunctor_i<N, double> {
public:
    virtual ~evalfunctor_i() { }
    virtual additive_gen_bto<N, btod_traits::bti_traits> &get_bto() = 0;

};


/** \brief Evaluates an expression that contains both tensors and
        operations (T = double)
    \tparam N Tensor order.

    An expression that consists of both individual tensors and
    sub-operations is evaluated as a sum of tensors (btod_add<N>) and
    a sum of operations (btod_sum<N>).

    \ingroup labeled_btensor_expr
 **/
template<size_t N>
class evalfunctor<N, double> : public evalfunctor_i<N, double> {
private:
    expr_rhs<N, double> &m_expr;
    eval_container_i<N, double> &m_eval_container;
    btod_copy<N> *m_op_copy;
    btod_add<N> *m_op_add;
    btod_sum<N> *m_op_sum;
    additive_gen_bto<N, btod_traits::bti_traits> *m_bto;
    btod_set<N> m_op_set;

public:
    /** \brief Initializes the functor
     **/
    evalfunctor(expr_rhs<N, double> &e, eval_container_i<N, double> &cont);

    /** \brief Virtual destructor
     **/
    virtual ~evalfunctor();

    /** \brief Returns the block tensor operation
     **/
    virtual additive_gen_bto<N, btod_traits::bti_traits> &get_bto() {
        if(m_bto == 0) make_bto();
        return *m_bto;
    }

    virtual btod_set<N> &get_clean_bto() {
        return m_op_set;
    }

private:
    void make_bto();
    void destroy_bto();

};


template<size_t N>
evalfunctor<N, double>::evalfunctor(
    expr_rhs<N, double> &e,
    eval_container_i<N, double> &cont) :

    m_expr(e), m_eval_container(cont),
    m_op_copy(0), m_op_add(0), m_op_sum(0), m_bto(0) {

}


template<size_t N>
evalfunctor<N, double>::~evalfunctor() {

    destroy_bto();
}


template<size_t N>
void evalfunctor<N, double>::make_bto() {

    destroy_bto();

    size_t ntensor = m_eval_container.get_ntensor();
    size_t noper = m_eval_container.get_noper();

    typedef arg<N, double, tensor_tag> targ_t;
    typedef arg<N, double, oper_tag> oarg_t;

    if(ntensor > 0 && noper > 0) {

        targ_t arg0 = m_eval_container.get_tensor_arg(0);
        m_op_add = new btod_add<N>(arg0.get_btensor(), arg0.get_perm(),
            arg0.get_coeff());
        for(size_t i = 1; i < ntensor; i++) {
            targ_t arg = m_eval_container.get_tensor_arg(i);
            m_op_add->add_op(arg.get_btensor(), arg.get_perm(),
                arg.get_coeff());
        }

        m_op_sum = new btod_sum<N>(*m_op_add, 1.0);
        for(size_t i = 0; i < noper; i++) {
            oarg_t arg = m_eval_container.get_oper_arg(i);
            m_op_sum->add_op(arg.get_operation(), arg.get_coeff());
        }

        m_bto = m_op_sum;

    } else if(ntensor > 1 && noper == 0) {

        targ_t arg0 = m_eval_container.get_tensor_arg(0);
        m_op_add = new btod_add<N>(arg0.get_btensor(), arg0.get_perm(),
            arg0.get_coeff());
        for(size_t i = 1; i < ntensor; i++) {
            targ_t arg = m_eval_container.get_tensor_arg(i);
            m_op_add->add_op(arg.get_btensor(), arg.get_perm(),
                arg.get_coeff());
        }

        m_bto = m_op_add;

    } else if(ntensor == 1 && noper == 0) {

        targ_t arg = m_eval_container.get_tensor_arg(0);
        m_op_copy = new btod_copy<N>(arg.get_btensor(), arg.get_perm(),
            arg.get_coeff());

        m_bto = m_op_copy;

    } else if(ntensor == 0 && noper > 0) {

        oarg_t arg0 = m_eval_container.get_oper_arg(0);
        m_op_sum = new btod_sum<N>(arg0.get_operation(), arg0.get_coeff());
        for(size_t i = 1; i < noper; i++) {
            oarg_t arg = m_eval_container.get_oper_arg(i);
            m_op_sum->add_op(arg.get_operation(), arg.get_coeff());
        }

        m_bto = m_op_sum;

    }
}


template<size_t N>
void evalfunctor<N, double>::destroy_bto() {

    delete m_op_sum; m_op_sum = 0;
    delete m_op_add; m_op_add = 0;
    delete m_op_copy; m_op_copy = 0;
    m_bto = 0;
}


} // namespace labeled_btensor_expr
} // namespace libtensor

#endif // LIBTENSOR_LABELED_BTENSOR_EXPR_EVALFUNCTOR_DOUBLE_H
