#ifndef LIBTENSOR_TOD_BTCONV_H
#define LIBTENSOR_TOD_BTCONV_H

#include <list>
#include <libtensor/timings.h>
#include <libtensor/tod/bad_dimensions.h>
#include <libtensor/tod/processor.h>
#include <libtensor/block_tensor/block_tensor_i.h>
#include "dense_tensor_i.h"

namespace libtensor {

/** \brief Unfolds a block tensor into a simple tensor
    \tparam N Tensor order.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_btconv : public timings< tod_btconv<N> > {
public:
    static const char *k_clazz; //!< Class name

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
        loop_list_node()
            : m_weight(0), m_inca(0), m_incb(0), m_op(NULL) { }
        loop_list_node(size_t weight, size_t inca, size_t incb)
            : m_weight(weight), m_inca(inca), m_incb(incb),
            m_op(NULL) { }
        processor_op_i_t *op() const { return m_op; }
    };

    class op_loop : public processor_op_i_t {
    private:
        size_t m_len, m_inca, m_incb;
    public:
        op_loop(size_t len, size_t inca, size_t incb)
            : m_len(len), m_inca(inca), m_incb(incb) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

    class op_dcopy : public processor_op_i_t {
    private:
        size_t m_len, m_inca, m_incb;
        double m_c;
    public:
        op_dcopy(size_t len, size_t inca, size_t incb, double c)
            : m_len(len), m_inca(inca), m_incb(incb), m_c(c) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

private:
    block_tensor_rd_i<N, double> &m_bt; //!< Source block %tensor

public:
    //!    \name Construction and destruction
    //@{

    tod_btconv(block_tensor_rd_i<N, double> &bt);
    ~tod_btconv();

    //@}

    //!    \name Tensor operation
    //@{

    void perform(dense_tensor_wr_i<N, double> &t);

    //@}

private:
    void copy_block(double *optr, const dimensions<N> &odims,
        const index<N> &ooffs, const double *iptr,
        const dimensions<N> &idims, const permutation<N> &iperm,
        double icoeff);
};


} // namespace libtensor

#endif // LIBTENSOR_TOD_BTCONV_H
