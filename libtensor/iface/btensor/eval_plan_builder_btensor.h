#ifndef LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H
#define LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H

#include <libtensor/expr/eval_plan.h>
#include <libtensor/iface/tensor_list.h>

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
    tensor_list &m_tl; //!< Tensor list
    expr::eval_plan m_plan; //!< Evaluation plan

public:
    eval_plan_builder_btensor(
        const expr::node_assign &node,
        tensor_list &tl) :

        m_assign(node), m_tl(tl)
    { }

    /** \brief Builds the evaluation plan
     **/
    void build_plan();

    /** \brief Returns the evaluation plan
     **/
    const expr::eval_plan &get_plan() const {
        return m_plan;
    }

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_PLAN_BUILDER_BTENSOR_H
