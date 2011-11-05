#ifndef LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_H
#define LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_H

#include "../../core/block_index_subspace_builder.h"
#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_proj_down.h"
#include "../se_part.h"
#include "partition_set.h"

namespace libtensor {


/**	\brief Implementation of so_proj_down<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_proj_down<N, M, T>, se_part<N, T> > :
public symmetry_operation_impl_base< so_proj_down<N, M, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_proj_down<N, M, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_proj_down<N, M, T>, se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_proj_down<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_proj_down<N, M, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter_t;

    //	Verify that the projection mask is correct
    //
    const mask<N> &m = params.msk;
    size_t nm = 0;
    for(size_t i = 0; i < N; i++) if(m[i]) nm++;
    if(nm != N - M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");
    }

    adapter_t g1(params.grp1);
    params.grp2.clear();

    // create block index space of result
    if (g1.is_empty()) return;

    typename adapter_t::iterator it = g1.begin();
    block_index_subspace_builder<N - M, M> rbb(g1.get_elem(it).get_bis(), m);

    partition_set<N, T> pset1(g1);
    partition_set<N - M, T> pset2(rbb.get_bis());
    // setup stabilizing masks
    mask<N> msk[1];
    for (size_t i = 0; i < N; i++) {
        if (! m[i]) msk[0][i] = true;
    }

    pset1.stabilize(msk, pset2);
    pset2.convert(params.grp2);

}

} // namespace libtensor

#endif // LIBTENSOR_SO_PROJ_DOWN_IMPL_PART_H
