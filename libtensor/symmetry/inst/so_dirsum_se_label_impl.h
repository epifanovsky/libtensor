#ifndef LIBTENSOR_SO_DIRSUM_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_DIRSUM_SE_LABEL_IMPL_H

#include "../combine_label.h"

namespace libtensor {

template<size_t N, size_t M, size_t NM, typename T>
const char *
symmetry_operation_impl< so_dirsum<N, M, T>, se_label<NM, T> >::k_clazz =
        "symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> >";

template<size_t N, size_t M, size_t NM, typename T>
void
symmetry_operation_impl< so_dirsum<N, M, T>, se_label<NM, T> >::do_perform(
    symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    // Adapter type for the input groups
    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_label<M, T> > adapter2_t;

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);
    params.g3.clear();

    sequence<N, size_t> map1(0);
    sequence<M, size_t> map2(0);
    { // map result index to input index
        sequence<N + M, size_t> map(0);
        for (size_t j = 0; j < N + M; j++) map[j] = j;
        permutation<N + M> pinv(params.perm, true);
        pinv.apply(map);

        for (size_t j = 0; j < N; j++) map1[j] = map[j];
        for (size_t j = 0; j < M; j++) map2[j] = map[j + N];
    }

    dimensions<N + M> bidims = params.bis.get_block_index_dims();

    //  Loop over each element in the first source group
    std::set<std::string> table_ids;
    for (typename adapter1_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        const se_label<N, T> &e1 = g1.get_elem(it1);
        if (table_ids.count(e1.get_table_id()) != 0) continue;

        combine_label<N, T> cl1(e1);
        table_ids.insert(cl1.get_table_id());

        { // Look for other se_label elements with the same table id
            typename adapter1_t::iterator it1b = it1; it1b++;
            for (; it1b != g1.end(); it1b++) {
                const se_label<N, T> &se1b = g1.get_elem(it1b);
                if (se1b.get_table_id() != cl1.get_table_id()) continue;
                cl1.add(se1b);
            }
        }

        // Create result se_label
        se_label<N + M, T> e3(bidims, cl1.get_table_id());
        transfer_labeling(cl1.get_labeling(), map1, e3.get_labeling());
        evaluation_rule<N + M> r3;

        // Look for an element in the second source group that has the
        // same product table
        typename adapter2_t::iterator it2 = g2.begin();
        for (; it2 != g2.end(); it2++) {
            if (e1.get_table_id() == g2.get_elem(it2).get_table_id()) break;
        }

        // If there is none
        if (it2 == g2.end()) {
            // Create a fake rule for non-existing e2 that allows all blocks
            sequence<N + M, size_t> seq3(0);
            for (register size_t i = 0; i < N; i++) seq3[map1[i]] = 1;

            product_rule<N + M> &pr3 = r3.new_product();
            pr3.add(seq3, product_table_i::k_invalid);
        }
        else {
            // If there is an se_label with the same product table
            combine_label<M, T> cl2(g2.get_elem(it2));

            { // Look for other se_label elements with the same table id
                typename adapter2_t::iterator it2b = it2; it2b++;
                for (; it2b != g2.end(); it2b++) {
                    const se_label<M, T> &se2b = g2.get_elem(it2b);
                    if (se2b.get_table_id() != cl2.get_table_id()) continue;
                    cl2.add(se2b);
                }
            }

            transfer_labeling(cl2.get_labeling(), map2, e3.get_labeling());
            e3.get_labeling().match();

            // Combine the evaluation rules
            // First transfer r1
            const evaluation_rule<N> &r1 = cl1.get_rule();

            for (typename evaluation_rule<N>::const_iterator ir1 = r1.begin();
                    ir1 != r1.end(); ir1++) {

                const product_rule<N> &pr1 = r1.get_product(ir1);
                if (pr1.empty()) continue;

                // Create new product in r3
                product_rule<N + M> &pr3 = r3.new_product();

                // Loop over terms in product of r1
                for (typename product_rule<N>::iterator ip1 = pr1.begin();
                        ip1 != pr1.end(); ip1++) {

                    sequence<N + M, size_t> seq3(0);
                    const sequence<N, size_t> &seq1 = pr1.get_sequence(ip1);
                    for (register size_t j = 0; j < N; j++)
                        seq3[map1[j]] = seq1[j];

                    pr3.add(seq3, ip1->second);
                }
            }

            // Then transfer r2
            const evaluation_rule<M> &r2 = cl2.get_rule();

            for (typename evaluation_rule<M>::const_iterator ir2 = r2.begin();
                    ir2 != r2.end(); ir2++) {

                const product_rule<M> &pr2 = r2.get_product(ir2);
                if (pr2.empty()) continue;

                // Create new product in r3
                product_rule<N + M> pr3 = r3.new_product();

                // Loop over terms in product of r2
                for (typename product_rule<M>::iterator ip2 = pr2.begin();
                        ip2 != pr2.end(); ip2++) {

                    sequence<N + M, size_t> seq3(0);
                    const sequence<M, size_t> &seq2 = pr2.get_sequence(ip2);
                    for (register size_t j = 0; j < M; j++)
                        seq3[map2[j]] = seq2[j];

                    pr3.add(seq3, ip2->second);
                }
            }
        }

        e3.set_rule(r3);
        params.g3.insert(e3);
    }

    // Now look in the second source group for symmetry elements that have not
    // been taken care off
    for (typename adapter2_t::iterator it2 = g2.begin();
            it2 != g2.end(); it2++) {

        const se_label<M, T> &e2 = g2.get_elem(it2);
        if (table_ids.count(e2.get_table_id()) != 0) continue;

        combine_label<M, T> cl2(e2);
        table_ids.insert(cl2.get_table_id());

        { // Look for other se_label elements with the same table id
            typename adapter2_t::iterator it2b = it2; it2b++;
            for (; it2b != g2.end(); it2b++) {
                const se_label<M, T> &se2b = g2.get_elem(it2b);
                if (se2b.get_table_id() != cl2.get_table_id()) continue;
                cl2.add(se2b);
            }
        }

        // Create result se_label
        se_label<N + M, T> e3(bidims, cl2.get_table_id());
        transfer_labeling(cl2.get_labeling(), map2, e3.get_labeling());

        evaluation_rule<N + M> r3;

        // Create a fake rule that allows all blocks
        sequence<N + M, size_t> seq3(0);
        for (register size_t i = 0; i < M; i++) seq3[map2[i]] = 1;

        product_rule<N + M> &pr3 = r3.new_product();
        pr3.add(seq3, product_table_i::k_invalid);

        // Set the rule and finish off
        e3.set_rule(r3);
        params.g3.insert(e3);
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_LABEL_IMPL_H
