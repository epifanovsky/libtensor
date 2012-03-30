#ifndef LIBTENSOR_TOD_IMPORT_RAW_H
#define LIBTENSOR_TOD_IMPORT_RAW_H

#include <list>
#include <libtensor/core/dimensions.h>
#include "dense_tensor_i.h"
#include <libtensor/tod/processor.h>
#include <libtensor/tod/bad_dimensions.h>

namespace libtensor {


/** \brief Imports %tensor elements from memory
    \tparam N Tensor order.

    This operation reads %tensor elements from a given window of a block
    of memory. The elements in the memory must be in the usual %tensor
    format. The block is characterized by its %dimensions, as if it were
    a part of the usual %tensor object. The window is specified by a range
    of indexes.

    The size of the recipient (result of the operation) must agree with
    the window dimensions.

    \ingroup libtensor_tod
 **/
template<size_t N>
class tod_import_raw {
public:
    static const char *k_clazz; //!< Class name

private:
    const double *m_ptr; //!< Pointer to data
    dimensions<N> m_dims; //!< Dimensions of the memory block
    index_range<N> m_ir; //!< Index range of the window

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
        size_t m_len;
    public:
        op_dcopy(size_t len) : m_len(len) { }
        virtual void exec(processor_t &proc, registers &regs)
            throw(exception);
    };

public:
    /**	\brief Initializes the operation
        \param ptr Pointer to data block
        \param dims Dimensions of the data block
        \param ir Index range of the window
    **/
    tod_import_raw(const double *ptr, const dimensions<N> &dims,
        const index_range<N> &ir) :
        m_ptr(ptr), m_dims(dims), m_ir(ir) { }

    /**	\brief Performs the operation
        \param t Output %tensor
    **/
    void perform(dense_tensor_i<N, double> &t);

};


} // namespace libtensor

#endif // LIBTENSOR_TOD_IMPORT_RAW_H
