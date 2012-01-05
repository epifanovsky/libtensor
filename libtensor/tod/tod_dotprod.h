#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include <list>
#include "../exception.h"
#include "../timings.h"
#include "../linalg/linalg.h"
#include "../core/permutation.h"
#include "../dense_tensor/dense_tensor_i.h"
#include "../core/tensor_ctrl.h"
#include "../mp/auto_cpu_lock.h"
#include "contraction2.h"
#include "bad_dimensions.h"
#include "processor.h"

namespace libtensor {


/**	\brief Calculates the dot product of two tensors
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_dotprod : public timings< tod_dotprod<N> > {
public:
    static const char *k_clazz; //!< Class name

private:
    struct registers {
        const double *m_ptra;
        const double *m_ptrb;
        double *m_ptrc;
    };

    struct loop_list_node;
    typedef std::list<loop_list_node> loop_list_t;
    typedef processor<loop_list_t,registers> processor_t;
    typedef processor_op_i<loop_list_t,registers> processor_op_i_t;

    struct loop_list_node {
    public:
        size_t m_weight;
        size_t m_inca, m_incb;
        processor_op_i_t *m_op;
        loop_list_node() :
            m_weight(0), m_inca(0), m_incb(0), m_op(NULL) {
        }
        loop_list_node(size_t weight, size_t inca, size_t incb) :
            m_weight(weight), m_inca(inca), m_incb(incb), m_op(NULL) {
        }
        processor_op_i_t *op() const {
            return m_op;
        }
    };

    class op_loop : public processor_op_i_t {
    private:
        size_t m_len, m_inca, m_incb;
    public:
        op_loop(size_t len, size_t inca, size_t incb) :
            m_len(len), m_inca(inca), m_incb(incb) {
        }
        virtual void exec(processor_t &proc, registers &regs) throw(exception);
    };

    class op_ddot : public processor_op_i_t, public timings<op_ddot> {
    private:
        size_t m_n, m_inca, m_incb;
    public:
        op_ddot(size_t n, size_t inca, size_t incb) :
            m_n(n), m_inca(inca), m_incb(incb) {
        }
        virtual void exec(processor_t &proc, registers &regs) throw(exception);

        static const char *k_clazz;
    };

private:
    dense_tensor_i<N,double> &m_t1; //!< First %tensor
    tensor_ctrl<N,double> m_tctrl1; //!< First %tensor control
    dense_tensor_i<N,double> &m_t2; //!< Second %tensor
    tensor_ctrl<N,double> m_tctrl2; //!< Second %tensor control
    permutation<N> m_perm1; //!< Permutation of the first %tensor
    permutation<N> m_perm2; //!< Permutation of the second %tensor
    loop_list_t m_list; //!< Loop list

public:
    //!	\name Construction and destruction
    //@{

    tod_dotprod(dense_tensor_i<N, double> &t1, dense_tensor_i<N, double> &t2);

    tod_dotprod(dense_tensor_i<N, double> &t1, const permutation<N> &perm1,
        dense_tensor_i<N, double> &t2, const permutation<N> &perm2);

    //@}

    /**	\brief Prefetches the arguments
     **/
    void prefetch();

    /**	\brief Computes the dot product
     **/
    double calculate(cpu_pool &cpus);

private:
    bool verify_dims();
    void clean_list();
    void build_list(loop_list_t &list, const dimensions<N> &da,
        const permutation<N> &pa, const dimensions<N> &db) throw(out_of_memory);

};


} // namespace libtensor


#ifndef LIBTENSOR_INSTANTIATE_TEMPLATES
#include "tod_dotprod_impl.h"
#endif // LIBTENSOR_INSTANTIATE_TEMPLATES

#endif // LIBTENSOR_TOD_DOTPROD_H
