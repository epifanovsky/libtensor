#ifndef LIBTENSOR_SO_DIRPROD_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_DIRPROD_SE_LABEL_IMPL_H

#include "../combine_label.h"

namespace libtensor {

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
    std::set<std::string> id_done;
    for (typename adapter1_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        const se_label<N, T> &e1 = g1.get_elem(it1);
        if (id_done.count(e1.get_table_id()) != 0) continue;

        combine_label<N, T> cl1(e1);
        id_done.insert(cl1.get_table_id());
        typename adapter1_t::iterator it1b = it1; it1b++;
        for (; it1b != g1.end(); it1b++) {
            const se_label<N, T> &se1b = g1.get_elem(it1b);
            if (se1b.get_table_id() != cl1.get_table_id()) continue;
            cl1.add(se1b);
        }

        // Create result se_label
        se_label<N + M, T> e3(bidims, cl1.get_table_id());
        transfer_labeling(cl1.get_labeling(), map1, e3.get_labeling());

        // Transfer the sequences
        evaluation_rule<N + M> r3;
        const evaluation_rule<N> &r1 = cl1.get_rule();

        std::map<size_t, size_t> m1to3;
        for (size_t i = 0; i < r1.get_n_sequences(); i++) {

            const sequence<N, size_t> &rs1 = r1[i];
            sequence<N + M, size_t> rs3(0);
            for (register size_t j = 0; j < N; j++) rs3[map1[j]] = rs1[j];

            m1to3[i] = r3.add_sequence(rs3);
        }

        // Look for an element in the second source group that has the
        // same product table
        typename adapter2_t::iterator it2 = g2.begin();
        for (; it2 != g2.end(); it2++) {

            if (e1.get_table_id() == g2.get_elem(it2).get_table_id())
                break;
        }

        // If there is none
        if (it2 == g2.end()) {

            // Transfer the products
            for (size_t i = 0; i < r1.get_n_products(); i++) {

                typename evaluation_rule<N>::iterator ip = r1.begin(i);
                size_t pno = r3.add_product(m1to3[r1.get_seq_no(ip)],
                        r1.get_intrinsic(ip), r1.get_target(ip));
                ip++;
                for (; ip != r1.end(i); ip++) {
                    r3.add_to_product(pno, m1to3[r1.get_seq_no(ip)],
                            r1.get_intrinsic(ip), r1.get_target(ip));
                }
            }
        }
        else {
            // If there is an se_label with the same product table
            combine_label<M, T> cl2(g2.get_elem(it2));
            typename adapter2_t::iterator it2b = it2; it2b++;
            for (; it2b != g2.end(); it2b++) {
                const se_label<M, T> &se2b = g2.get_elem(it2b);
                if (se2b.get_table_id() != cl2.get_table_id()) continue;
                cl2.add(se2b);
            }
            transfer_labeling(cl2.get_labeling(), map2, e3.get_labeling());
            e3.get_labeling().match();

            // First transfer the sequences
            const evaluation_rule<M> &r2 = cl2.get_rule();

            std::map<size_t, size_t> m2to3;
            for (size_t i = 0; i < r2.get_n_sequences(); i++) {

                const sequence<M, size_t> &rs2 = r2[i];
                sequence<N + M, size_t> rs3(0);
                for (register size_t j = 0; j < M; j++) rs3[map2[j]] = rs2[j];

                m2to3[i] = r3.add_sequence(rs3);
            }

            // Then merge the products
            for (size_t i = 0; i < r1.get_n_products(); i++) {

                for (size_t j = 0; j < r2.get_n_products(); j++) {

                    typename evaluation_rule<N>::iterator ip1 = r1.begin(i);
                    size_t pno = r3.add_product(m1to3[r1.get_seq_no(ip1)],
                            r1.get_intrinsic(ip1), r1.get_target(ip1));
                    ip1++;

                    for (; ip1 != r1.end(i); ip1++)
                        r3.add_to_product(pno, m1to3[r1.get_seq_no(ip1)],
                                r1.get_intrinsic(ip1), r1.get_target(ip1));

                    for (typename evaluation_rule<M>::iterator ip2 =
                            r2.begin(j); ip2 != r2.end(j); ip2++)
                        r3.add_to_product(pno, m2to3[r2.get_seq_no(ip2)],
                                r2.get_intrinsic(ip2), r2.get_target(ip2));
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
        if (id_done.count(e2.get_table_id()) != 0) continue;

        combine_label<M, T> cl2(e2);
        id_done.insert(cl2.get_table_id());
        typename adapter2_t::iterator it2b = it2; it2b++;
        for (; it2b != g2.end(); it2b++) {
            const se_label<M, T> &se2b = g2.get_elem(it2b);
            if (se2b.get_table_id() != cl2.get_table_id()) continue;
            cl2.add(se2b);
        }

        // Create result se_label
        se_label<N + M, T> e3(bidims, cl2.get_table_id());
        transfer_labeling(cl2.get_labeling(), map2, e3.get_labeling());

        // Transfer the rule from e2
        evaluation_rule<N + M> r3;
        const evaluation_rule<M> &r2 = cl2.get_rule();

        std::map<size_t, size_t> m2to3;
        for (size_t i = 0; i < r2.get_n_sequences(); i++) {

             const sequence<M, size_t> &rs2 = r2[i];
             sequence<N + M, size_t> rs3(0);
             for (register size_t j = 0; j < M; j++) rs3[map2[j]] = rs2[j];

             m2to3[i] = r3.add_sequence(rs3);
        }

        // Transfer the products
        for (size_t i = 0; i < r2.get_n_products(); i++) {

            typename evaluation_rule<M>::iterator ip = r2.begin(i);
            size_t pno = r3.add_product(m2to3[r2.get_seq_no(ip)],
                    r2.get_intrinsic(ip), r2.get_target(ip));
            ip++;
            for (; ip != r2.end(i); ip++) {
                r3.add_to_product(pno, m2to3[r2.get_seq_no(ip)],
                        r2.get_intrinsic(ip), r2.get_target(ip));
            }
        }

        // Set the rule and finish off
        e3.set_rule(r3);
        params.g3.insert(e3);
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_SE_LABEL_IMPL_H
