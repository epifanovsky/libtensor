#ifndef LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H
#define LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H

#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"
#include "../combine_part.h"

namespace libtensor {

template<size_t N, typename T>
const char *symmetry_operation_impl< so_symmetrize<N, T>,
se_part<N, T> >::k_clazz =
        "symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_symmetrize<N, T>, se_part<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    combine_part<N, T> cp(params.grp1);
    se_part<N, T> sp1a(cp.get_bis(), cp.get_pdims());
    cp.perform(sp1a);
    se_part<N, T> sp1b(sp1a);
    sp1b.permute(params.perm);


    if (sp1b.get_pdims() != cp.get_pdims()) {
        throw bad_symmetry(g_ns, k_clazz, method,
                __FILE__, __LINE__, "Incompatible dimensions.");
    }

    se_part<N, T> sp2(cp.get_bis(), cp.get_pdims());

    abs_index<N> ai(cp.get_pdims());
    do {

        const index<N> &i1 = ai.get_index();
        if (sp1a.is_forbidden(i1) && sp1b.is_forbidden(i1)) {
            sp2.mark_forbidden(i1);
            continue;
        }

        if (sp1a.is_forbidden(i1)) {
            const index<N> &i1b = sp1b.get_direct_map(i1);
            if (sp1a.is_forbidden(i1b)) {
                sp2.add_map(i1, i1b, sp1b.get_sign(i1, i1b));
            }
        }
        else if (sp1b.is_forbidden(i1)) {
            const index<N> &i1a = sp1a.get_direct_map(i1);
            if (sp1b.is_forbidden(i1a)) {
                sp2.add_map(i1, i1a, sp1a.get_sign(i1, i1a));
            }
        }
        else {
            const index<N> &i1a = sp1a.get_direct_map(i1);
            if (sp1b.map_exists(i1, i1a)) {
                bool sign = sp1a.get_sign(i1, i1a);
                if (sign == sp1b.get_sign(i1, i1a)) {
                    sp2.add_map(i1, i1a, sign);
                }
            }
            const index<N> &i1b = sp1b.get_direct_map(i1);
            if (sp1a.map_exists(i1, i1b)) {
                bool sign = sp1b.get_sign(i1, i1b);
                if (sign == sp1a.get_sign(i1, i1b)) {
                    sp2.add_map(i1, i1b, sign);
                }
            }
        }
    } while (ai.inc());

    params.grp2.insert(sp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE_SE_PART_IMPL_H
