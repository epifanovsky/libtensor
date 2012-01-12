#ifndef LIBTENSOR_SO_REDUCE_IMPL_PART_H
#define LIBTENSOR_SO_REDUCE_IMPL_PART_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/block_index_subspace_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_reduce.h"
#include "../se_part.h"
#include "combine_part.h"

namespace libtensor {


/**	\brief Implementation of so_reduce<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_reduce<N, M, K, T>, se_part<N, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, K, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order2 = N - M; //!< Dimension of result

public:
    typedef so_reduce<N, M, K, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, size_t K, typename T>
const char *symmetry_operation_impl< so_reduce<N, M, K, T>, se_part<N, T> >
::k_clazz = "symmetry_operation_impl< so_reduce<N, M, K, T>, se_part<N, T> >";

template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_reduce<N, M, K, T>, se_part<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, element_t> adapter_t;
    typedef se_part<k_order2, T> el2_t;
    typedef std::list< index<N> > ilist_t;

    //	Verify that the projection masks are correct
    //
    mask<N> tm, rm;
    size_t nm = 0;
    for (size_t k = 0; k < K; k++) {
        const mask<N> &m = params.msk[k];
        for(size_t i = 0; i < N; i++) {
            if(! m[i]) continue;
            if(tm[i]) {
                throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                        "params.msk[k]");
            }

            tm[i] = true;
            nm++;
        }
    }
    if(nm != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
                "params.msk");
    }

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create a map of indexes
    sequence<N, size_t> map(0);
    for (size_t i = 0, j = 0; i < N; i++) {
        if (tm[i]) {
            for (size_t k = 0; k < K; k++) {
                if (! params.msk[k][i]) continue;
                map[i] = k_order2 + k;
            }
        }
        else {
            map[i] = j++;
        }
    }

    combine_part<N, T> cp(params.grp1);
    element_t el1(cp.get_bis(), cp.get_pdims());
    cp.perform(el1);

    const dimensions<N> &pdims1 = el1.get_pdims();

    index<k_order2> ia, ib;
    index<K> ixa, ixb;
    for (size_t i = 0; i < N; i++) {
        rm[i] = ! tm[i];
        if (tm[i]) {
            ixb[map[i] - k_order2] = pdims1[i] - 1;
        }
        else {
            ib[map[i]] = pdims1[i] - 1;
        }
    }

    dimensions<k_order2> pdims2(index_range<k_order2>(ia, ib));
    block_index_subspace_builder<k_order2, M> bb(el1.get_bis(), rm);
    el2_t el2(bb.get_bis(), pdims2);

    bool empty = true;

    dimensions<K> pdimsx(index_range<K>(ixa, ixb));

    abs_index<k_order2> ai2a(pdims2);
    do {
        const index<k_order2> &i2a = ai2a.get_index();

        // Create a list of all possible indexes from the input
        ilist_t la;
        abs_index<K> ai3a(pdimsx);
        do {
            const index<K> &i3a = ai3a.get_index();

            index<N> i1a;
            for (size_t i = 0, j = 0; i < N; i++) {
                if (tm[i]) i1a[i] = i3a[map[i] - k_order2];
                else i1a[i] = i2a[j++];
            }

            if (! el1.is_forbidden(i1a)) la.push_back(i1a);

        } while (ai3a.inc());

        if (la.empty()) {
            el2.mark_forbidden(i2a);
            empty = false;
            continue;
        }

        abs_index<k_order2> ai2b(i2a, pdims2);
        while (ai2b.inc()) {

            const index<k_order2> &i2b = ai2b.get_index();

            ilist_t lb;

            abs_index<K> ai3b(pdimsx);
            do {
                const index<K> &i3b = ai3b.get_index();

                index<N> i1b;
                for (size_t i = 0, j = 0; i < N; i++) {
                    if (tm[i]) i1b[i] = i3b[map[i] - k_order2];
                    else i1b[i] = i2b[j++];
                }

                if (! el1.is_forbidden(i1b)) lb.push_back(i1b);

            } while (ai3b.inc());

            if (lb.empty()) continue;

            bool found = false, sign = true;
            typename ilist_t::iterator ila = la.begin();
            for ( ; ila != la.end(); ila++) {

                typename ilist_t::iterator ilb = lb.begin();
                for ( ; ilb != lb.end(); ilb++) {
                    if (el1.map_exists(*ila, *ilb)) break;
                } // for lb

                if (ilb == lb.end()) break;

                bool s = el1.get_sign(*ila, *ilb);
                if (found && sign != s) break;

                sign = s;
                found = true;
                lb.erase(ilb);
            } // for la
            if (ila != la.end()) continue;

            el2.add_map(i2a, i2b, sign);
            empty = false;
            break;

        } // while ai2b
    } while (ai2a.inc());

    if (! empty) params.grp2.insert(el2);
}


} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_IMPL_PART_H
