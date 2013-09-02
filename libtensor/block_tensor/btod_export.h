#ifndef LIBTENSOR_BTOD_EXPORT_H
#define LIBTENSOR_BTOD_EXPORT_H

#include <libtensor/core/dimensions.h>
#include <libtensor/core/noncopyable.h>
#include <libtensor/block_tensor/block_tensor_i.h>

namespace libtensor {


/** \brief Unfolds a block tensor into a data array
    \tparam N Tensor order.

    \ingroup libtensor_btod
 **/
template<size_t N>
class btod_export : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    typedef block_tensor_i_traits<double> bti_traits;

private:
    gen_block_tensor_rd_i<N, bti_traits> &m_bt; //!< Source block %tensor

public:
    /** \brief Constructs the operation
        \param bt Block tensor.
     **/
    btod_export(gen_block_tensor_rd_i<N, bti_traits> &bt);

    /** \brief Virtual destructor
     **/
    virtual ~btod_export();

    /** \brief Copies block tensor into an array
        \param ptr Data pointer.
     **/
    void perform(double *ptr);

private:
    void copy_block(double *optr, const dimensions<N> &odims,
        const index<N> &ooffs, const double *iptr,
        const dimensions<N> &idims, const permutation<N> &iperm,
        double icoeff);

};


} // namespace libtensor

#endif // LIBTENSOR_BTOD_EXPORT_H
