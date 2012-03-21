#ifndef LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H
#define LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_dirsum.h"
#include "../se_label.h"

namespace libtensor {


/**	\brief Implementation of so_dirsum<N, M, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> > :
public symmetry_operation_impl_base< so_dirsum<N, M, T>, se_label<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirsum<N, M, T> operation_t;
    typedef se_label<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> >";


template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_dirsum<N, M, T>, se_label<N + M, T> >::do_perform(
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
    //  Loop over each element in the first source group
    for (typename adapter1_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        const se_label<N, T> &e1 = g1.get_elem(it1);

        // Create result se_label
        se_label<N + M, T> e3(bidims, e1.get_table_id());
        transfer_labeling(e1.get_labeling(), map1, e3.get_labeling());

        // Look for an element in the second source group that has the
        // same product table
        typename adapter2_t::iterator it2 = g2.begin();
        for (; it2 != g2.end(); it2++) {

            if (e1.get_table_id() == g2.get_elem(it2).get_table_id())
                break;
        }

        evaluation_rule<N + M> r3;

        // If there is none
        if (it2 == g2.end()) {

            // Create a fake rule for non-existing e2 that allows all blocks
            sequence<N + M, size_t> seq3(1);

            size_t seqno = r3.add_sequence(seq3);
            r3.add_product(seqno, product_table_i::k_invalid);
        }
        else {
            // If there is an se_label with the same product table
            const se_label<M, T> &e2 = g2.get_elem(it2);
            transfer_labeling(e2.get_labeling(), map2, e3.get_labeling());

            // Combine the evaluation rules

            // First transfer r1
            const evaluation_rule<N> &r1 = e1.get_rule();
            std::map<size_t, size_t> m1to3;
            for (size_t i = 0; i < r1.get_n_sequences(); i++) {

                const sequence<N, size_t> &rs1 = r1[i];
                sequence<N + M, size_t> rs3(0);
                for (size_t j = 0; j < N; j++) rs3[map1[j]] = rs1[j];

                m1to3[i] = r3.add_sequence(rs3);
            }
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

            // Then transfer r2
            const evaluation_rule<M> &r2 = e2.get_rule();

            std::map<size_t, size_t> m2to3;
            for (size_t i = 0; i < r2.get_n_sequences(); i++) {

                const sequence<M, size_t> &rs2 = r2[i];
                sequence<N + M, size_t> rs3(0);
                for (size_t j = 0; j < M; j++) rs3[map2[j]] = rs2[j];

                m2to3[i] = r3.add_sequence(rs3);
            }

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
        }

        e3.set_rule(r3);
        params.g3.insert(e3);
    }

    // Now look in the second source group for symmetry elements that have not
    // been taken care off
    for (typename adapter2_t::iterator it2 = g2.begin();
            it2 != g2.end(); it2++) {

        const se_label<M, T> &e2 = g2.get_elem(it2);

        typename adapter1_t::iterator it1 = g1.begin();
        for (; it1 != g1.end(); it1++) {

            if (g1.get_elem(it1).get_table_id() == e2.get_table_id()) break;
        }
        if (it1 != g1.end()) continue;

        // Create result se_label
        se_label<N + M, T> e3(bidims, e2.get_table_id());
        transfer_labeling(e2.get_labeling(), map2, e3.get_labeling());

        // Create a fake rule that allows all blocks
        evaluation_rule<N + M> r3;
        sequence<N + M, size_t> seq3(1);

        size_t seqno = r3.add_sequence(seq3);
        r3.add_product(seqno, product_table_i::k_invalid);

        // Set the rule and finish off
        e3.set_rule(r3);
        params.g3.insert(e3);
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H
