#ifndef LIBTENSOR_TOD_DIAG_H
#define LIBTENSOR_TOD_DIAG_H

#include <list>
#include <libtensor/linalg/linalg.h>
#include <libtensor/timings.h>
#include <libtensor/core/mask.h>
#include <libtensor/core/permutation.h>
#include <libtensor/core/scalar_transf_double.h>
#include <libtensor/core/tensor_transf.h>
#include <libtensor/tod/processor.h>
#include "dense_tensor_i.h"

namespace libtensor {


/** \brief Extracts a general diagonal from a %tensor
    \tparam N Tensor order.
    \tparam M Diagonal order.

    Extracts a general multi-dimensional diagonal from a %tensor. The
    diagonal to extract is specified by a %mask, unmasked indexes remain
    intact. The order of the result is (n-m+1), where n is the order of
    the original %tensor, m is the order of the diagonal.

    The order of indexes in the result is the same as in the argument with
    the exception of the collapsed diagonal. The diagonal's index in the
    result correspond to the first its index in the argument, for example:
    \f[ c_i = a_{ii} \qquad c_{ip} = a_{iip} \qquad c_{ip} = a_{ipi} \f]
    The specified permutation may be applied to the result to alter the
    order of the indexes.

    A coefficient (default 1.0) is specified to scale the elements along
    with the extraction of the diagonal.

    If the number of set bits in the %mask is not equal to M, the %mask
    is incorrect, which causes a \c bad_parameter exception upon the
    creation of the operation. If the %dimensions of the output %tensor
    are wrong, the \c bad_dimensions exception is thrown.

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class tod_diag : public timings< tod_diag<N, M> > {
public:
    static const char *k_clazz; //!< Class name

public:
    static const size_t k_ordera = N; //!< Order of the source %tensor
    static const size_t k_orderb =
        N - M + 1; //!< Order of the destination %tensor

    typedef tensor_transf<k_orderb, double> tensor_transf_type;

private:
    struct registers {
        const double *m_ptra;
        double *m_ptrb;
    };

    struct loop_list_node;
    typedef std::list<loop_list_node> loop_list_t;
    typedef processor<loop_list_t, registers> processor_t;
    typedef processor_op_i<loop_list_t, registers> processor_op_i_t;

    struct loop_list_node {
    public:
        size_t m_weight;
        size_t m_inca, m_incb;
        processor_op_i_t *m_op;
        loop_list_node() :
            m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
        loop_list_node(size_t weight, size_t inca, size_t incb) :
            m_weight(weight), m_inca(inca), m_incb(incb),
            m_op(NULL) { }
        processor_op_i_t *op() const { return m_op; }
    };

    class op_loop : public processor_op_i_t {
    private:
        size_t m_len, m_inca, m_incb;
    public:
        op_loop(size_t len, size_t inca, size_t incb) :
            m_len(len), m_inca(inca), m_incb(incb) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

    class op_dcopy : public processor_op_i_t, public timings<op_dcopy> {
    public:
        static const char *k_clazz; //!< Class name
    private:
        size_t m_len, m_inca, m_incb;
        double m_c;
    public:
        op_dcopy(size_t len, size_t inca, size_t incb, double c) :
            m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

    class op_daxpy : public processor_op_i_t, public timings<op_daxpy> {
    public:
        static const char *k_clazz; //!< Class name
    private:
        size_t m_len, m_inca, m_incb;
        double m_c;
    public:
        op_daxpy(size_t len, size_t inca, size_t incb, double c)
            : m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

private:
    dense_tensor_rd_i<N, double> &m_t; //!< Input %tensor
    mask<N> m_mask; //!< Diagonal mask
    tensor_transf_type m_tr; //!< Transformation of the result
    dimensions<k_orderb> m_dims; //!< Dimensions of the result

public:
    /** \brief Creates the operation
        \param t Input %tensor.
        \param m Diagonal mask.
        \param p Permutation of result.
        \param c Scaling coefficient (default 1.0)
     **/
    tod_diag(dense_tensor_rd_i<N, double> &t, const mask<N> &m,
            const tensor_transf_type &tr = tensor_transf_type());

    /** \brief Performs the operation, adds to the output
        \param zero Zero result first
        \param c Scalar transformation to apply before adding to result
        \param tb Output %tensor.
     **/
    void perform(bool zero, dense_tensor_wr_i<k_orderb, double> &tb);

private:
    /** \brief Forms the %dimensions of the output or throws an
        exception if the input is incorrect
     **/
    static dimensions<N - M + 1> mk_dims(
        const dimensions<N> &dims, const mask<N> &msk);

    /** \brief Forms the loop and executes the operation
     **/
    template<typename CoreOp>
    void do_perform(dense_tensor_wr_i<k_orderb, double> &tb);

    /** \brief Builds the nested loop list
     **/
    template<typename CoreOp>
    void build_list(loop_list_t &list, dense_tensor_wr_i<k_orderb, double> &tb);

    /** \brief Cleans the nested loop list
     **/
    void clean_list(loop_list_t &list);

private:
    tod_diag(const tod_diag&);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_DIAG_H
