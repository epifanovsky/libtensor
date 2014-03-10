#ifndef LIBTENSOR_CTF_TOD_SCALE_H
#define LIBTENSOR_CTF_TOD_SCALE_H

#include <libtensor/core/noncopyable.h>
#include <libtensor/core/scalar_transf_double.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Scales a distributed tensor by a constant
    \tparam N Tensor order.

    \sa tod_scale

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_scale : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    double m_c; //!< Scaling coefficient

public:
    /** \brief Initializes the operation
        \param c Scaling coefficient.
     **/
    ctf_tod_scale(const scalar_transf<double> &c);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_tod_scale() { }

    /** \brief Performs the operation
        \param ta Tensor.
     **/
    void perform(ctf_dense_tensor_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SCALE_H
