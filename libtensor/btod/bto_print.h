#ifndef LIBTENSOR_BTO_PRINT_H
#define LIBTENSOR_BTO_PRINT_H

#include <ostream>
#include <iomanip>
#include <libtensor/defs.h>
#include <libtensor/exception.h>
#include <libtensor/core/allocator.h>
#include <libtensor/dense_tensor/dense_tensor.h>
#include <libtensor/dense_tensor/dense_tensor_ctrl.h>
#include <libtensor/dense_tensor/to_btconv.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {

template<size_t N, typename T, typename Alloc = allocator >
class bto_print {
public:
    static const char *k_clazz; //!< Class name

private:
    std::ostream &m_stream; //!< Output stream
    size_t m_precision; //!< Precision of the output
    bool m_sci; //!< Set the scientific flag on output

public:
    bto_print(std::ostream &stream, size_t pre = 16, bool sci = true) :
        m_stream(stream), m_precision(pre), m_sci(sci) { }

    void perform(block_tensor_rd_i<N,T> &bt);
};

template<size_t N, typename T, typename Alloc>
const char *bto_print<N, T, Alloc>::k_clazz = "bto_print<N, T, Alloc>";


template<size_t N, typename T, typename Alloc>
void bto_print<N, T, Alloc>::perform(block_tensor_rd_i<N, T> &bt)
{
    static const char *method = "perform(block_tensor_rd_i<N, T>&)";

    if(! m_stream.good()) {
        throw_exc(k_clazz, method, "Output stream not initialized.");
    }

    const dimensions<N> &dims = bt.get_bis().get_dims();
    m_stream << N;
    for (size_t i = 0; i < N; i++) {
        m_stream << " " << dims.get_dim(i);
    }
    m_stream << std::endl;

    dense_tensor<N, T, Alloc> ta(dims);
    to_btconv<N, T>(bt).perform(ta);

    dense_tensor_ctrl<N, T> ctrla(ta);
    const T *cptra = ctrla.req_const_dataptr();

    size_t width;
    if (m_sci)
        width = m_precision + 9;
    else
        width = m_precision + 6;

    for (size_t i=0; i < dims.get_size(); i++) {
        m_stream << std::setw(width) << std::setprecision(m_precision);
        m_stream << std::right;
        if ( m_sci )
            m_stream << std::scientific;
        else
            m_stream << std::fixed;

        m_stream << cptra[i];

        if (i % 3 == 2)
            m_stream << std::endl;
        else
            m_stream << " ";
    }
    m_stream << std::endl;

    ctrla.ret_const_dataptr(cptra);
}


} // namespace libtensor


#endif // LIBTENSOR_BTO_PRINT_H
