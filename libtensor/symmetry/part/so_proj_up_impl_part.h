#ifndef LIBTENSOR_SO_PROJ_UP_IMPL_PART_H
#define LIBTENSOR_SO_PROJ_UP_IMPL_PART_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../not_implemented.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_proj_up.h"
#include "../se_part.h"
#include "partition_set.h"

namespace libtensor {


/**	\brief Implementation of so_proj_up<N, M, T> for se_perm<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_proj_up<N, M, T>, se_part<N, T> > :
public symmetry_operation_impl_base<
so_proj_up<N, M, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_proj_up<N, M, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_proj_up<N, M, T>,
se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_proj_up<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_proj_up<N, M, T>,
se_part<N, T> >::do_perform(symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    // Adapter type for the input group
    typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter_t;

    //	Verify that the mask is valid
    const mask<N + M> &m = params.msk;
    size_t nm = 0;
    for(size_t i = 0; i < N + M; i++) if(m[i]) nm++;
    if(nm != N) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");
    }

    adapter_t g1(params.g1);
    params.g2.clear();

    // map  result index -> input index
    sequence<N, size_t> map(0);
    for (size_t j = 0; j < N; j++) map[j] = j;
    params.perm.apply(map);

    // create partition set of result
    partition_set<N + M, T> pset(params.bis);

    const permutation<N> &perm = params.perm;

    //	Go over each element in the first source group
    for(typename adapter_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_part<N, T> &e1 = g1.get_elem(i);

        // Add e1 to partition set
        pset.add_partition(e1, perm, m);
    }

    pset.convert(params.g2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_UP_IMPL_LABEL_H
