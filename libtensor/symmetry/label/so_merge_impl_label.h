#ifndef LIBTENSOR_SO_MERGE_IMPL_LABEL_H
#define LIBTENSOR_SO_MERGE_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_merge.h"
#include "../se_label.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	This implementation sets the target label to all labels.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> > :
public symmetry_operation_impl_base< so_merge<N, M, K, T>, se_label<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, M, K, T> operation_t;
    typedef se_label<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, size_t M, size_t K, typename T>
const char *symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >
::k_clazz = "symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >";


template<size_t N, size_t M, size_t K, typename T>
void symmetry_operation_impl< so_merge<N, M, K, T>, se_label<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    static const size_t k_orderc = N - M + K;

    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef se_label<k_orderc, T> el2_t;

    //	Verify that the projection masks are correct
    size_t nm = 0;
    mask<N> tm;
    for (size_t k = 0; k < K; k++) {
        const mask<N> &m = params.msk[k];
        for (size_t i = 0; i < N; i++) {
            if (! m[i]) continue;
            if (tm[i]) {
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

    sequence<N, size_t> map;
    for (size_t i = 0, j = 0; i < N; i++) {
        // Check if this is the first masked index in one of the masks
        if (tm[i]) {
            size_t k = 0;
            for ( ; k < K; k++) { if (params.msk[k][i]) break; }

            const mask<N> &m = params.msk[k];
            for (k = 0; k < i; k++) {  if (m[k]) break; }
            if (k != i) { map[i] = map[k]; continue; }
        }

        map[i] = j++;
    }

    sequence<M, size_t> nmap;
    for (register size_t i = 0; i < N; i++) nmap[map[i]]++;

    adapter_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 = g1.get_elem(it1).get_block_index_dims();

    index<k_orderc> idx1, idx2;
    for (size_t i = 0, j = 0; i < N; i++) {
        if (! mm[i]) continue;
        idx2[j++] = bidims1[i] - 1;
    }

    dimensions<k_orderc> bidims2(index_range<k_orderc>(idx1, idx2));
    // Check the bidims for correctness
    for (size_t i = 0; i < N; i++) {
        if (bidims2[map[i]] != bidims1[i]) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "bidims2");
        }
    }

    std::list<el2_t> lst2;

    // Loop over all se_label elements and merge dimensions in each one
    for (; it1 != g1.end(); it1++) {

#ifdef LIBTENSOR_DEBUG
        // This should never happen!!!
        if (bidims1 != g1.get_elem(it1).get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method,
                    __FILE__, __LINE__, "Incompatible se_labels in input.");
        }
#endif

        const element_t &se1 = g1.get_elem(it1);
        el2_t se2(bidims2, se1.get_table_id());

        // Loop over all
        const composite_rule &r1 = se1.get_rule();
        composite_rule r2(r1, map);
        for (it = r1.begin(); it != r1.end(); it++) {


            r2.push_back(it);

        }


        // Loop over all label sets
        for (typename element_t::const_iterator iss = se1.begin();
                itl != se1.end(); iss++) {

            const typename element_t::set_t &ss1 = se1.get_subset(itl);
            typename el2_t::set_t &ss2 = se2.create_subset(ss1.get_table_id());

            // Transfer evaluation mask and block labels
            const mask<N> &emsk1 = ss1.get_mask();
            mask<M> emsk2;

            for (size_t i = 0; i < N; i++) {

                emsk2[map[i]] = (emsk2[map[i]] != emsk1[i]);

                mask<M> dtmsk;
                dtmsk[map[i]] = true;
                size_t itype1 = ss1.get_dim_type(i);
                size_t itype2 = ss2.get_dim_type(i);

                for (size_t ipos = 0; ipos < bidims1[i]; ipos++) {

                    typename element_t::set_t::label_t l1 =
                            ss1.get_label(itype, ipos);

                    if (! ss1.is_valid(l1)) continue;

                    typename el2_t::set_t::label_t l2 =
                            ss2.get_label(itype, ipos);
                    if (ss2.is_valid(l2)) {
                        if (l1 != l2) {
                            throw bad_symmetry(g_ns, k_clazz, method,
                                    __FILE__, __LINE__, "Illegal labeling.");
                        }
                    }
                    else {
                        ss2.assign(dtmsk, ipos, l1);
                    }
                } // for ipos
            } // for i

            ss2.set_mask(emsk2);
            // Check if the intrinsic labels need to change:
            for (size_t k = 0; k < K; k++) {

            }

            // Transfer intrinsic labels
            for (typename element_t::set_t::iterator ii = ss1.begin();
                    ii != ss1.end(); ii++) {
                ss2.add_intrinsic(ss1.get_intrinsic(ii));
            }



            ss2.match_blk_labels();
        } // for iss

        // Check if the same se_label already exists in the list
        typename std::list<el2_t>::const_iterator ilst2 = lst2.begin();
        for (; ilst2 != lst2.end(); ilst2++) {

            const el2_t &se2b = *ilst2;

            // Loop over all label sets in se2
            typename el2_t::const_iterator iss2a = se2.begin();
            for (; iss2a != se2.end(); iss2a++) {

                const typename el2_t::set_t &ss2a = se2.get_subset(iss2a);
                const std::string &t2a = ss2a.get_table_id();
                const mask<k_orderc> &m2a = ss2a.get_mask();

                // Loop over all label sets in se2b
                typename el2_t::const_iterator iss2b = se2b.begin();
                for (; iss2b != se2b.end(); iss2b++) {
                    const typename el2_t::set_t &ss2b = se2b.get_subset(iss2b);
                    if (ss2b.get_table_id() == t2a
                            && ss2b.get_mask() == m2a) break;
                }
                // Label set not found
                if (iss2b != se2b.end()) break;
            }
            // Some label set were not found
            if (iss2a != se2.end()) break;
        }

        // se_label not found
        if (ilst2 == lst2.end()) {
            lst2.push_back(se2);
            continue;
        }

        // Combine se2 and se2b
        const el2_t &se2b = *ilst2;
        // Loop over all label sets in se2
        for (iss2a = se2.begin(); iss2a != se2.end(); iss2a++) {

            const typename el2_t::set_t &ss2a = se2.get_subset(iss2a);
            const std::string &t2a = ss2a.get_table_id();
            const mask<k_orderc> &m2a = ss2a.get_mask();

            // Find the respective label set in se2b
            typename el2_t::iterator iss2b = se2b.begin();
            for (; iss2b != se2b.end(); iss2b++) {
                const typename el2_t::set_t &ss2b = se2b.get_subset(iss2b);
                if (ss2b.get_table_id() == t2a
                        && ss2b.get_mask() == m2a) break;
            }

            typename el2_t::set_t &ss2b = se2b.get_subset(iss2b);

            // Compile a list of intrinsic labels that are in both sets
            std::list<typename el2_t::set_t::label_t> intr;
            for (typename el2_t::set_t::iterator iila = ss2a.begin();
                    iila != ss2a.end(); iila++) {

                typename el2_t::set_t::label_t ila = ss2a.get_intrinsic(iila);

                typename el2_t::set_t::iterator iilb = ss2b.begin();
                for (; iilb != ss2b.end(); iilb++) {

                    if (ila == ss2b.get_intrinsic(iilb)) break;
                }

                if (iilb == ss2b.end()) continue;

                intr.push_back(ila);
            }

            // Delete intrinsic labels in ss2b and replace them by the list
            ss2b.clear_intrinsic();
            for (std::list<typename el2_t::set_t::label_t>::iterator ii =
                    intr.begin(); ii != intr.end(); ii++) {

                ss2b.add_intrinsic(*ii);
            }
        } // Loop iss2a

    } // Loop it1

    // At last put all se_label elements in to the result group
    for (std::list<el2_t>::iterator it = lst2.begin();
            it != lst2.end(); it++) {

        params.grp2.insert(*it);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
