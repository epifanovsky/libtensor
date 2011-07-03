#ifndef LIBTENSOR_SO_DIRSUM_IMPL_PART_H
#define LIBTENSOR_SO_DIRSUM_IMPL_PART_H

#include <map>
#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_dirsum.h"
#include "../se_part.h"
#include "combine_part.h"

namespace libtensor {


/**	\brief Implementation of so_dirsum<N, M, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirsum<N, M, T>, se_part<N + M, T> > :
public symmetry_operation_impl_base< so_dirsum<N, M, T>, se_part<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirsum<N, M, T> operation_t;
    typedef se_part<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_dirsum<N, M, T>,
se_part<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirsum<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>,
se_part<N + M, T> >::do_perform(symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    params.g3.clear();

    // Merge symmetry element set 1 into one se_part
    combine_part<N, T> p1(params.g1);
    se_part<N, T> se1(p1.get_bis(), p1.get_pdims());
    p1.perform(se1);

    // Merge symmetry element set 2 into one se_part
    combine_part<M, T> p2(params.g2);
    se_part<M, T> se2(p2.get_bis(), p2.get_pdims());
    p2.perform(se2);

    // Map where which input index ends up in the output
    sequence<N + M, size_t> map(0);
    for (size_t i = 0; i < N + M; i++) map[i] = i;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    // Build the partition dimensions of the result
    index<N + M> i3a, i3b;
    dimensions<N> pdims1 = se1.get_pdims();
    for (size_t i = 0; i < N; i++) i3b[map[i]] = pdims1[i] - 1;
    dimensions<M> pdims2 = se2.get_pdims();
    for (size_t i = 0; i < M; i++) i3b[map[i + N]] = pdims2[i] - 1;

    // Construct the result
    se_part<N + M, T> se3(params.bis,
            dimensions<N + M>(index_range<N + M>(i3a, i3b)));

    // Loop over all result indexes
    abs_index<N + M> aix(se3.get_pdims());
    do {
        const index<N + M> &ix = aix.get_index();

        index<N> i1a;
        for (size_t i = 0; i < N; i++) i1a[i] = ix[map[i]];
        index<M> i2a;
        for (size_t i = 0; i < M; i++) i2a[i] = ix[map[i + N]];

        bool forbidden1 = se1.is_forbidden(i1a);
        bool forbidden2 = se2.is_forbidden(i2a);

        // Result partitions are forbidden, if the respective partitions in
        // both input elements are forbidden
        if (forbidden1 && forbidden2) {
            se3.mark_forbidden(ix);
        }
        else if (forbidden1) {
            index<M> i2b = se2.get_direct_map(i2a);
            if (i2a == i2b) continue;

            for (size_t i = 0; i < N; i++) i3b[map[i]] = i1a[i];
            for (size_t i = 0; i < M; i++) i3b[map[i + N]] = i2b[i];

            se3.add_map(ix, i3b, se2.get_sign(i2a, i2b));
        }
        else if (forbidden2) {
            index<N> i1b = se1.get_direct_map(i1a);
            if (i1a == i1b) continue;

            for (size_t i = 0; i < N; i++) i3b[map[i]] = i1b[i];
            for (size_t i = 0; i < M; i++) i3b[map[i + N]] = i2a[i];

            se3.add_map(ix, i3b, se1.get_sign(i1a, i1b));
        }
        else {

            // Get direct maps
            index<N> i1b = se1.get_direct_map(i1a);
            index<M> i2b = se2.get_direct_map(i2a);
            //
            bool sign = se1.get_sign(i1a, i1b);
            if (sign != se2.get_sign(i2a, i2b)) continue;

            for (size_t i = 0; i < N; i++) i3b[map[i]] = i1b[i];
            for (size_t i = 0; i < M; i++) i3b[map[i + N]] = i2b[i];

            se3.add_map(ix, i3b, sign);
        }
    } while (aix.inc());

    params.g3.insert(se3);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_IMPL_PART_H
