#ifndef LIBTENSOR_TO_DIAG_DIMS_H
#define LIBTENSOR_TO_DIAG_DIMS_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/dimensions.h>
#include <libtensor/core/mask.h>

namespace libtensor {


/** \brief Computes the dimensions of the output of tod_diag
    \tparam N Tensor order.
    \tparam M Diagonal order.

    \sa tod_diag

    \ingroup libtensor_tod
 **/
template<size_t N, size_t M>
class to_diag_dims : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

public:
    enum {
        NA = N,
        NB = M
    };

private:
    dimensions<NB> m_dimsb; //!< Dimensions of the result

public:
    /** \brief Computes the dimensions
        \param dimsa Dimensions of input tensor.
        \param m Diagonal mask.
        \param permb Permutation of result.
     **/
    to_diag_dims(
        const dimensions<NA> &dimsa,
        const sequence<NA, size_t> &m,
        const permutation<NB> &permb);

    /** \brief Returns the dimensions of the output
     **/
    const dimensions<NB> &get_dimsb() const {
        return m_dimsb;
    }

};


} // namespace libtensor

#endif // LIBTENSOR_TO_DIAG_DIMS_H
