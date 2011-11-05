#ifndef LIBTENSOR_SO_SYMMETRIZE3_IMPL_PERM_H
#define LIBTENSOR_SO_SYMMETRIZE3_IMPL_PERM_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_symmetrize3.h"
#include "../se_perm.h"
#include "permutation_group.h"

namespace libtensor {


/**	\brief Implementation of so_symmetrize3<N, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_symmetrize3<N, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_symmetrize3<N, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_symmetrize3<N, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_symmetrize3<N, T>,
se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize3<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize3<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter_t;

    adapter_t adapter1(params.grp1);
    permutation_group<N, T> grp2(adapter1);
    permutation<N> p1(params.pperm), p2(params.pperm);
    p2.permute(params.cperm);
    grp2.add_orbit(params.symm, p1);
    grp2.add_orbit(params.symm, p2);

    params.grp2.clear();
    grp2.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE3_IMPL_PERM_H
