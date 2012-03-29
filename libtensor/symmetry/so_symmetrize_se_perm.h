#ifndef LIBTENSOR_SO_SYMMETRIZE_IMPL_PERM_H
#define LIBTENSOR_SO_SYMMETRIZE_IMPL_PERM_H

#include "se_perm.h"
#include "so_symmetrize.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/**	\brief Implementation of so_symmetrize<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_symmetrize<N, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_symmetrize<N, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_symmetrize<N, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

private:
    static size_t encode(const permutation<N> &p);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_IMPL_PERM_H
