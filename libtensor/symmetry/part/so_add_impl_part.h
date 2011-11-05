#ifndef LIBTENSOR_SO_ADD_IMPL_PART_H
#define LIBTENSOR_SO_ADD_IMPL_PART_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_add.h"
#include "../se_part.h"
#include "partition_set.h"

namespace libtensor {


/**	\brief Implementation of so_add<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_add<N, T>, se_part<N, T> > :
public symmetry_operation_impl_base< so_add<N, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_add<N, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_add<N, T>, se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_add<N, T>, se_part<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_add<N, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, element_t > adapter_t;
    adapter_t g1(params.grp1);
    adapter_t g2(params.grp2);
    params.grp3.clear();

    if (g1.is_empty() || g2.is_empty()) return;

    partition_set<N, T> p1(g1, params.perm1);
    partition_set<N, T> p2(g2, params.perm2);

    p1.intersect(p2, false);
    p1.convert(params.grp3);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_ADD_IMPL_LABEL_H
