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

    permutation<N> q0, q1, q2, q3, q4, q5;
    q1.permute(params.cperm);
    q2.permute(q1).permute(q1);
    q3.permute(params.pperm);
    q4.permute(q3).permute(q1);
    q5.permute(q3).permute(q2);

    combine_part<N, T> cp(params.grp1);
    se_part<N, T> sp1a(cp.get_bis(), cp.get_pdims());
    cp.perform(sp1a);
    se_part<N, T> sp1b(sp1a), sp1c(sp1a), sp1d(sp1a), sp1e(sp1a), sp1f(sp1a);
    sp1b.permute(q1);
    sp1c.permute(q2);
    sp1d.permute(q3);
    sp1e.permute(q4);
    sp1f.permute(q5);

    se_part<N, T> *sp1[6];
    sp1[0] = &sp1a;
    sp1[1] = &sp1b;
    sp1[2] = &sp1c;
    sp1[3] = &sp1d;
    sp1[4] = &sp1e;
    sp1[5] = &sp1f;

    for (register size_t i = 0; i < 6; i++) {
        if (sp1[i]->get_pdims() != cp.get_pdims())
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible dimensions.");
    }

    se_part<N, T> sp2(cp.get_bis(), cp.get_pdims());

    abs_index<N> ai(cp.get_pdims());
    do {

        const index<N> &i1 = ai.get_index();
        bool all_forbidden = true;
        mask<6> forbidden;
        for (register size_t i = 0; i < 6; i++) {
            if (sp1[i].is_forbidden(i1)) continue;

            all_forbidden = false;

            const index<N> &i2 = sp1[i]->get_direct_map(i1);
            bool sign = sp1[i]->get_sign(i1, i2);
            register size_t j = 0;
            for (; j < 6; j++) {
                if (i == j) continue;
                if (sp1[i]->is_forbidden(i1) != sp1[i]->is_forbidden(i2)) break;
                if (! sp1[i]->map_exists(i1, i2)) break;
                if (sp1[i]->get_sign(i1, i2) != sign) break;
            }
            if (j == 6) {
                sp2.add_map(i1, i2, sign);
            }
        }
        if (all_forbidden) {
            sp2.mark_forbidden(i1);
        }
    } while (ai.inc());

    params.grp2.clear();
    p0.convert(params.grp2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE3_IMPL_PART_H
