#ifndef LIBTENSOR_SO_SYMMETRIZE3_IMPL_PART_H
#define LIBTENSOR_SO_SYMMETRIZE3_IMPL_PART_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_symmetrize3.h"
#include "../se_part.h"
#include "partition_set.h"

namespace libtensor {


/**	\brief Implementation of so_symmetrize3<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_symmetrize3<N, T>, se_part<N, T> > :
public symmetry_operation_impl_base< so_symmetrize3<N, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_symmetrize3<N, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_symmetrize3<N, T>,
se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize3<N, T>, se_part<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize3<N, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter_t;

    adapter_t g1(params.grp1);
    partition_set<N, T> p0(g1), p1(g1), p2(g1), p3(g1), p4(g1), p5(g1);
    permutation<N> q0, q1, q2, q3, q4, q5;

    q1.permute(params.cperm);
    q2.permute(q1).permute(q1);
    q3.permute(params.pperm);
    q4.permute(q3).permute(q1);
    q5.permute(q3).permute(q2);

    p1.permute(q1);
    p2.permute(q2);
    p3.permute(q3);
    p4.permute(q4);
    p5.permute(q5);

    p0.intersect(p1);
    p0.intersect(p2);
    p0.intersect(p3);
    p0.intersect(p4);
    p0.intersect(p5);

    params.grp2.clear();
    p0.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE3_IMPL_PART_H
