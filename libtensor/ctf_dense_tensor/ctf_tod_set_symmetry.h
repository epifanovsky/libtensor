#ifndef LIBTENSOR_CTF_TOD_SET_SYMMETRY_H
#define LIBTENSOR_CTF_TOD_SET_SYMMETRY_H

#include <libtensor/core/noncopyable.h>
#include "ctf_dense_tensor_i.h"
#include "ctf_symmetry.h"

namespace libtensor {


/** \brief Sets or adjusts the permutational symmetry of a distributed tensor
    \tparam N Tensor order.

    This operation adjusts the permutational symmetry of a distributed tensor
    to a subgroup of its current symmetry.

    \ingroup libtensor_ctf_dense_tensor
 **/
template<size_t N>
class ctf_tod_set_symmetry : public noncopyable {
public:
    static const char k_clazz[]; //!< Class name

private:
    ctf_symmetry<N, double> m_sym; //!< Symmetry

public:
    /** \brief Initializes the operation
        \param sym New symmetry
     **/
    ctf_tod_set_symmetry(const ctf_symmetry<N, double> &sym) :
        m_sym(sym)
    { }

    /** \brief Performs the operation
        \param zero If true, reset current symmetry to new symmetry
        \param ta Tensor
     **/
    void perform(bool zero, ctf_dense_tensor_i<N, double> &ta);

};


} // namespace libtensor

#endif // LIBTENSOR_CTF_TOD_SET_SYMMETRY_H
