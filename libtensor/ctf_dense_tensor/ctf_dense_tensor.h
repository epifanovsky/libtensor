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
    ctf_symmetry<N, T> m_sym; //!< Tensor symmetry
    tCTF_Tensor<double> *m_tens; //!< CTF tensor

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
    /** \brief Returns the CTF tensor object
     **/
    virtual tCTF_Tensor<T> &on_req_ctf_tensor();

    /** \brief Handles requests for the symmetry of the CTF tensor
     **/
    virtual const ctf_symmetry<N, T> &on_req_symmetry();

    /** \brief Resets the symmetry of the CTF tensor
     **/
    virtual void on_reset_symmetry(const ctf_symmetry<N, T> &sym);

    /** \brief Called when state changes to immutable
     **/
    virtual void on_set_immutable();

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_DENSE_TENSOR_H
