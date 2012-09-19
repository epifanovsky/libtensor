#ifndef LIBTENSOR_TOD_SCATTER_H
#define LIBTENSOR_TOD_SCATTER_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Scatters a lower-order tensor in a higher-order tensor
    \tparam N Order of the first tensor.
    \tparam M Order of the result less the order of the first tensor.

    Given a tensor \f$ a_{ij\cdots} \f$, the operation computes
    \f$ c_{\cdots ij\cdots} = k_a a_{ij\cdots} \f$.

    The order of tensor indexes in the result can be specified using
    a permutation.

    \sa tod_dirsum

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_scatter : public timings< tod_scatter<N, M> > {
public:
    static const char *k_clazz; //!< Class name

public:
    enum {
        k_ordera = N, //!< Order of the first %tensor
        k_orderc = N + M //!< Order of the result
    };

    typedef tensor_transf<k_orderc, double> tensor_transf_type;

private:
    struct registers {
        const double *m_ptra;
        double *m_ptrc;
    };

    struct loop_list_node;
    typedef std::list<loop_list_node> loop_list_t;
    typedef typename loop_list_t::iterator loop_list_iterator_t;

    struct loop_list_node {
    public:
        size_t m_weight;
        size_t m_inca, m_incc;
        void (tod_scatter<N, M>::*m_fn)(registers &);
        loop_list_node() : m_weight(0), m_inca(0), m_incc(0), m_fn(0)
            { }
        loop_list_node(size_t weight, size_t inca, size_t incc) :
            m_weight(weight), m_inca(inca), m_incc(incc), m_fn(0)
            { }
    };

    //!    c_ji = a_i
    struct {
        double m_kc;
        size_t m_n;
        size_t m_stepc;
    } m_scatter;

private:
    dense_tensor_rd_i<k_ordera, double> &m_ta; //!< First tensor (A)
    tensor_transf_type m_trc; //!< Tensor transformation of result
    loop_list_t m_list; //!< Loop list

public:
    /** \brief Initializes the operation
        \param ta Input tensor
        \param trc Tensor transformation applied to result
     **/
    tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta,
            tensor_transf_type &trc) :
        m_ta(ta), m_trc(trc)
    { }

    /** \brief Initializes the operation
     **/
    tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta, double ka) :
        m_ta(ta), m_trc(permutation<k_orderc>(), scalar_transf<double>(ka))
    { }

    /** \brief Initializes the operation
     **/
    tod_scatter(dense_tensor_rd_i<k_ordera, double> &ta, double ka,
        const permutation<k_orderc> &permc) :
        m_ta(ta), m_trc(permc, scalar_transf<double>(ka))
    { }

    /** \brief Performs the operation
        \param zero Zero result first
     **/
    void perform(bool zero, dense_tensor_wr_i<k_orderc, double> &tc);

private:
    void check_dimsc(dense_tensor_wr_i<k_orderc, double> &tc);

private:
    void exec(loop_list_iterator_t &i, registers &regs);
    void fn_loop(loop_list_iterator_t &i, registers &regs);
    void fn_scatter(registers &regs);

private:
    tod_scatter(const tod_scatter&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_SCATTER_H

