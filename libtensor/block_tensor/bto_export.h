#ifndef LIBTENSOR_BTO_EXPORT_H
#define LIBTENSOR_BTO_EXPORT_H

#include <libtensor/core/dimensions.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {


/** \brief Unfolds a block tensor into a data array
    \tparam N Tensor order.

    \ingroup libtensor_bto
 **/
template<size_t N, typename T>
class bto_export : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef block_tensor_i_traits<T> bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bt; //!< Source block %tensor

public:
    /** \brief Constructs the operation
        \param bt Block tensor.
     **/
    bto_export(gen_block_tensor_rd_i<N, bti_traits> &bt);

    /** \brief Virtual destructor
     **/
    virtual ~bto_export();

    /** \brief Copies block tensor into an array
        \param ptr Data pointer.
     **/
    void perform(T *ptr);

private:
    void copy_block(T *optr, const dimensions<N> &odims,
        const index<N> &ooffs, const T *iptr,
        const dimensions<N> &idims, const permutation<N> &iperm,
        T icoeff);

};

template<size_t N>
using btod_export = bto_export<N, double>;

} // namespace libtensor

#endif // LIBTENSOR_BTO_EXPORT_H
