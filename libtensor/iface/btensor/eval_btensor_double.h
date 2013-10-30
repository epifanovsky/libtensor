#ifndef LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
#define LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H

#include <libtensor/expr/eval_plan.h>
#include <libtensor/iface/tensor_list.h>

namespace libtensor {
namespace iface {


/** \brief Processor of evaluation plan for btensor result type (double)

    \ingroup libtensor_iface
 **/
template<>
class eval_btensor<double> {
public:
    enum {
        Nmax = 8
    };

    typedef expr::node::tid_t tid_t; //!< Tensor ID type

public:
    /** \brief Processes an evaluation plan
        \param plan Evaluation plan.
     **/
    void process_plan(const expr::eval_plan &plan, tensor_list &tl);

private:
    void handle_assign(const expr::node_assign &node, tensor_list &tl);
    void handle_create_interm(tid_t tid, tensor_list &tl);
    void handle_delete_interm(tid_t tid, tensor_list &tl);

    void verify_tensor_type(tid_t tid, const tensor_list &tl);

};


} // namespace iface
} // namespace libtensor


#endif // LIBTENSOR_IFACE_EVAL_BTENSOR_DOUBLE_H
