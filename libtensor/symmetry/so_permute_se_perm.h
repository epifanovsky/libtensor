#ifndef LIBTENSOR_SO_PERMUTE_SE_PERM_H
#define LIBTENSOR_SO_PERMUTE_SE_PERM_H

#include "se_perm.h"
#include "so_permute.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/** \brief Implementation of so_permute<N, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_permute<N, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_permute<N, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_permute<N, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

public:
    virtual ~symmetry_operation_impl() { }

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};

} // namespace libtensor

#endif // LIBTENSOR_SO_PERMUTE_SE_PERM_H
