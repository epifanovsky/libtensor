#ifndef LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H
#define LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../../core/permutation_builder.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_dirprod.h"
#include "../se_label.h"
#include "label_set.h"
#include "transfer_label_set.h"

namespace libtensor {


/**	\brief Implementation of so_dirprod<N, M, T> for se_label<N + M, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> > :
public symmetry_operation_impl_base<
so_dirprod<N, M, T>, se_label<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirprod<N, M, T> operation_t;
    typedef se_label<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >";


template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirprod<N, M, T>, se_label<N + M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    // Adapter type for the input groups
    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_label<M, T> > adapter2_t;

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);
    params.g3.clear();

    // map result index to input index
    sequence<N + M, size_t> map(0);
    for (size_t j = 0; j < N + M; j++) map[j] = j;
    permutation<N + M> pinv(params.perm, true);
    pinv.apply(map);

    sequence<N, size_t> map1(0);
    for (size_t j = 0; j < N; j++) map1[j] = map[j];
    sequence<M, size_t> map2(0);
    for (size_t j = 0; j < M; j++) map2[j] = map[j + N];

    dimensions<N + M> bidims = params.bis.get_block_index_dims();
    //	Go over each element in the first source group
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_label<N, T> &e1 = g1.get_elem(i);
        const block_labeling<N> &bl1 = e1.get_labeling();

        // Create result se_label
        se_label<N + M, T> e3(bidims, e1.get_table_id());
        block_labeling<N + M> &bl3 = e3.get_labeling();

        // Look for an element in the second source group that has the
        // same product table
        typename adapter2_t::iterator j = g2.begin();
        for(; j != g2.end(); j++) {

            if (e1.get_table_id().compare(g2.get_elem(j).get_table_id()) == 0)
                break;
        }

        const se_label<M, T> &e2 = g2.get_elem(j);

        // Combine the block labelings


        // Combine the evaluation rules
        const evaluation_rule &r1 = e1.get_rule();
        const evaluation_rule &r2 = e2.get_rule();
        evaluation_rule r3;

        typedef evaluation_rule::rule_iterator rule_iterator;
        typedef std::map<rule_iterator, rule_iterator> rule_iterator_map;
        typedef std::pair<rule_iterator, rule_iterator> rule_iterator_pair;
        rule_iterator_map m1to3, m2to3;

        for (rule_iterator ir = r1.begin(); ir != r1.end(); ir++) {
            const evaluation_rule::basic_rule &br1 = r1.get_rule(ir);
            rule_iterator ir3 = r3.add_rule(br1.intr, br1.order);
            m1to3.push_back(rule_iterator_pair(ir, ir3));
        }

        for (rule_iterator ir = r2.begin(); ir != r2.end(); ir++) {
            const evaluation_rule::basic_rule &br2 = r2.get_rule(ir);
            rule_iterator ir3 = r3.add_rule(br2.intr, br2.order);
            m2to3.push_back(rule_iterator_pair(ir, ir3));
        }


        for (sum_iterator is1 = r1.sbegin(); is1 != r1.send(); is1++) {
            for (sum_iterator is2 = r2.sbegin(); is2 != r2.send(); is2++) {

                product_iterator ip1 = r1.pbegin(is1);
                const evaluation_rule::rules_product &pr3 =
                        r3.new_product(m1to3[r1.get_rule(ip1)]);
                ip1++;

                for (; ip1 != r1.pend(is1); ip1++) {
                    r3.add_to_product(pr3, m1to3[r1.get_rule(ip1)]);
                }

                for (product_iterator ip2 = r.begin(); ip2 != pr2.end(); ip2++) {
                    r3.add_to_product(pr3, m2to3[r2.get_rule(ip2)]);
                }
            }
        }

        params.g3.insert(e3);
    }

    //  Do the same for the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        // Create result se_label
        se_label<N + M, T> e3(bidims);

        const se_label<M, T> &e2 = g2.get_elem(i);
        transfer_label_set<M, T>(e2).perform(map2, e3);

        params.g3.insert(e3);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_IMPL_LABEL_H
