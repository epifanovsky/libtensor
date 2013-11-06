#ifndef LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H
#define LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H

#include <libtensor/expr/eval_plan.h>
#include <libtensor/iface/tensor_list.h>
#include "interm.h"

namespace libtensor {
namespace iface {


class eval_plan_builder_btensor {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        Nmax = 8
    };

    typedef expr::node::tid_t tid_t; //!< Tensor ID type

private:
    expr::node_assign m_assign; //!< Assignment node
    tensor_list m_tl; //!< Tensor list
    expr::eval_plan m_plan; //!< Evaluation plan
    interm m_interm; //!< Intermediates store

public:
    eval_plan_builder_btensor(
        const expr::node_assign &node,
        const tensor_list &tl) :

        m_assign(node), m_tl(tl), m_interm(m_tl)
    { }

    /** \brief Builds the evaluation plan
     **/
    void build_plan();

    /** \brief Returns the tensor list
     **/
    const tensor_list &get_tensors() const {
        return m_tl;
    }

    /** \brief Returns the evaluation plan
     **/
    const expr::eval_plan &get_plan() const {
        return m_plan;
    }

    /** \brief Returns the intermediates container
     **/
    const interm &get_interm() const {
        return m_interm;
    }

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H
