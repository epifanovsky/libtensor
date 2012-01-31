#ifndef LIBTENSOR_TOD_DOTPROD_H
#define LIBTENSOR_TOD_DOTPROD_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/mp/cpu_pool.h>
#include "dense_tensor_i.h"

#include <libtensor/tod/processor.h>

namespace libtensor {


/**	\brief Calculates the inner (dot) product of two tensors
    \tparam N Tensor order.

    The inner (dot) product of two tensors is defined as
    \f$ d = sum_{ijk...} a_{ijk...} b_{ijk...} \f$

    \ingroup libtensor_dense_tensor_tod
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
    dense_tensor_i<N,double> &m_ta; //!< First tensor (A)
    dense_tensor_i<N,double> &m_tb; //!< Second tensor (B)
    permutation<N> m_perma; //!< Permutation of the first tensor (A)
    permutation<N> m_permb; //!< Permutation of the second tensor (B)
    loop_list_t m_list; //!< Loop list

public:
    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param tb Second tensor (B)
     **/
    tod_dotprod(dense_tensor_i<N, double> &ta, dense_tensor_i<N, double> &tb);

    /** \brief Initializes the operation
        \param ta First tensor (A)
        \param perma Permutation of first tensor (A)
        \param tb Second tensor (B)
        \param permb Permutation of second tensor (B)
     **/
    tod_dotprod(dense_tensor_i<N, double> &ta, const permutation<N> &perma,
        dense_tensor_i<N, double> &tb, const permutation<N> &permb);

    /**	\brief Prefetches the arguments
     **/
    void prefetch();

    /**	\brief Computes the dot product and returns the value
        \param cpus CPU pool.
     **/
    double calculate(cpu_pool &cpus);

private:
    bool verify_dims();
    void clean_list();
    void build_list(loop_list_t &list, const dimensions<N> &da,
        const permutation<N> &pa, const dimensions<N> &db) throw(out_of_memory);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_DOTPROD_H
