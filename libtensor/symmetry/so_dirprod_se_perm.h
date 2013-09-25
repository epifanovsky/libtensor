#ifndef LIBTENSOR_SO_DIRPROD_SE_PERM_H
#define LIBTENSOR_SO_DIRPROD_SE_PERM_H

#include "se_perm.h"
#include "so_dirprod.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {


/** \brief Implementation of so_dirprod<N, M, T> for se_perm<N + M, T>
    \tparam N Tensor order 1.
    \tparam M Tensor order 2.
    \tparam T Tensor element type.
    \tparam CGT Cyclic group traits

    Constructs the direct product of two permutation %symmetry groups.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirprod<N, M, T>, se_perm<N + M, T> > :
public symmetry_operation_impl_base< so_dirprod<N, M, T>, se_perm<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirprod<N, M, T> operation_t;
    typedef se_perm<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

public:
    virtual ~symmetry_operation_impl() { }

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_SE_PERM_H
