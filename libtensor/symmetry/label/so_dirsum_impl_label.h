#ifndef LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H
#define LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_dirsum.h"
#include "../se_label.h"
#include "transfer_label_set.h"

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

    params.g3.clear();
    if (params.g1.is_empty() && params.g2.is_empty()) return;

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);

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

    // Loop over each element in the first source group
    for(typename adapter1_t::iterator i = g1.begin(); i != g1.end(); i++) {

        const se_label<N, T> &e1 = g1.get_elem(i1);
        const std::string &id = e1.get_table_id();

        element_t e3(bidims, id);

        // First transfer block labels from e1 to e3
        const block_labeling<N> &bl1 = e1.get_labels();
        block_labeling<N + M> &l3 = e3.get_block_labeling();

        // Loop over all types in e1
        // Create a mask for every type
        // Assign labels to this type


        // Next transfer the evaluation rule
        // Loop over basic rules
        // Copy basic rule and add it result rule
        composite_rule r3;
        const composite_rule &r1 = e1.get_rule();
        for (i = 0; i < r1.size(); i++) {
            basic_rule b3(r1[i]);
            for (j = 0; j < b3.size(); j++) {
                if (b3[j] >= N) continue;

                b3[j] = map1[b3[j]];
            }

            r3.append(b3);
        }

        for (rit1 = r1.begin(); rit1 != r1.end(); r1++) {
            r3.append(r1.get_list(rit1));
        }

        //
        for (rit2 = r2.begin(); rit2 != r2.end(); rit2++) {
            rule_no_list p3(r2.get_list(rit2));
            for (pit = p3.begin(); pit != p3.end(); pit++) {
                *pit += r1.size();
            }
            r3.append(p3);
        }




        //
        for (rit1 = r1.begin(); rit1 != r1.end(); rit1++) {
            const rule_no_list &p1 = r1.get_list(rit1);
            for (rit2 = r2.begin(); rit2 != r2.end(); rit2++) {
                const rule_no_list &p2 = r2.get_list(rit2);

                rule_no_list p3(p1);
                pit = p3.end();
                p3.insert(pit, p2.begin(), p2.end());
                for ( ; pit != p3.end(); pit++) {
                    *pit += r1.size();
                }
                r3.append(p3);
            }
        }



        // Loop over each element in the second source group
        typename adapter2_t::iterator j = g2.begin();
        for( ; j != g2.end(); j++) {

            if (e1.get_table_id().compare(g2.get_elem(j).get_table_id()) == 0)
                break;
        }


        composite_rule r3;
        if (j == g2.end()) {
            //
        }
        else {
            // Loop over all typed in e2 ...

            //
            composite_rule
        }
    }

    // Loop over each element in the second source group
    for(typename adapter2_t::iterator i = g2.begin(); i != g2.end(); i++) {

        // Create template for result se_label in list l2
        l2.push_back(element_t(bidims));
        element_t &e3 = l2.back();

        const se_label<M, T> &e2 = g2.get_elem(i);
        transfer_label_set<M, T>(e2).perform(map2, e3);
    }

    if (l1.empty()) {

        // If l1 is empty copy all elements in l2, but extend all evaluation
        // masks to the l1 indexes before
        for (typename std::list<element_t>::iterator it = l2.begin();
                it != l2.end(); it++) {

            element_t &el2 = *it;
            for (typename element_t::iterator iss = el2.begin();
                    iss != el2.end(); iss++) {

                label_set<N + M> &ss2 = el2.get_subset(iss);
                mask<N + M> msk = ss2.get_mask();
                for (size_t j = 0; j < N; j++) msk[map[j]] = true;
                ss2.set_mask(msk);
            }

            params.g3.insert(el2);
        }
    }
    else if (l2.empty()) {

        // If l2 is empty copy all elements in l1, but extend all evaluation
        // masks to the l2 indexes before
        for (typename std::list<element_t>::iterator it = l1.begin();
                it != l1.end(); it++) {

            element_t &el1 = *it;
            for (typename element_t::iterator iss = el1.begin();
                    iss != el1.end(); iss++) {

                label_set<N + M> &ss1 = el1.get_subset(iss);
                mask<N + M> msk = ss1.get_mask();
                for (size_t j = 0; j < M; j++) msk[map[j + N]] = true;
                ss1.set_mask(msk);
            }
            params.g3.insert(el1);
        }
    }
    else {
        // Otherwise combine l1 and l2
        for (typename std::list<element_t>::iterator it1 = l1.begin();
                it1 != l1.end(); it1++) {

            element_t &e1 = *it1;

            for (typename std::list<element_t>::iterator it2 = l2.begin();
                    it2 != l2.end(); it2++) {

                element_t &e2 = *it2;

                element_t e3(e1);
                for (typename element_t::iterator iss = e2.begin();
                        iss != e2.end(); iss++) {

                    e3.add_subset(e2.get_subset(iss));
                }

                params.g3.insert(e3);
            }
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_IMPL_LABEL_H
