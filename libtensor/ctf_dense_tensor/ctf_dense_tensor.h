#ifndef LIBTENSOR_CTF_DENSE_TENSOR_H
#define LIBTENSOR_CTF_DENSE_TENSOR_H

#include <vector>
#include <libtensor/timings.h>
#include <libtensor/core/immutable.h>
#include <libtensor/core/noncopyable.h>
#include "ctf_dense_tensor_i.h"

namespace libtensor {


/** \brief Distributed tensor in Cyclops Tensor Framework
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N, typename T>
class ctf_dense_tensor :
    public ctf_dense_tensor_i<N, T>,
    public immutable,
    public noncopyable {

public:
    static const char k_clazz[]; //!< Class name

public:
    typedef T element_t; //!< Tensor element type

private:
    dimensions<N> m_dims; //!< Tensor dimensions
    int m_tid; //!< Tensor ID in CTF

public:
    /** \brief Creates a new tensor
        \param dims Non-zero tensor dimensions.
     **/
    ctf_dense_tensor(const dimensions<N> &dims);

    /** \brief Virtual destructor
     **/
    virtual ~ctf_dense_tensor();

    /** \brief Returns the dimensions of the tensor
     **/
    virtual const dimensions<N> &get_dims() const;

protected:
    /** \brief Returns tensor ID in CTF
     **/
    virtual int on_req_tensor_id();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_H
