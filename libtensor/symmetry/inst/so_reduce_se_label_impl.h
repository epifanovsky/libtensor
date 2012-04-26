#ifndef LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H
#define LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H

#include <map>
#include <set>
#include <libtensor/defs.h>
#include <libtensor/core/abs_index.h>
#include "../bad_symmetry.h"
#include "../combine_label.h"
#include "../product_table_container.h"

namespace libtensor {

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >::k_clazz =
        "symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >";

template<size_t N, size_t M, typename T>
void
symmetry_operation_impl< so_reduce<N, M, T>, se_label<N - M, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter<k_order1, T, el1_t> adapter_t;
    typedef product_table_i::label_t label_t;
    typedef product_table_i::label_group_t label_group_t;
    typedef product_table_i::label_set_t label_set_t;

    params.grp2.clear();
    if (params.grp1.is_empty()) return;

    // Create a map of remaining indexes
    sequence<k_order1, size_t> map((size_t) -1);
    size_t nrsteps = 0;
    for (register size_t i = 0, j = 0; i < k_order1; i++) {
        if (params.msk[i]) nrsteps = std::max(nrsteps, params.rseq[i]);
        else map[i] = j++;
    }
    nrsteps++;

    adapter_t g1(params.grp1);

    // Create block index dimensions of result se_label
    typename adapter_t::iterator it1 = g1.begin();
    const dimensions<N> &bidims1 =
            g1.get_elem(it1).get_labeling().get_block_index_dims();

    index<k_order2> idx1, idx2;
    for (size_t i = 0, j = 0; i < k_order1; i++) {
        if (params.msk[i]) {
#ifdef LIBTENSOR_DEBUG
            if (params.rblrange.get_end()[i] >= bidims1[i]) {
                throw bad_parameter(g_ns, k_clazz, method,
                        __FILE__, __LINE__, "rblrange.");
            }
#endif
        }
        else {
            idx2[j++] = bidims1[i] - 1;
        }
    }
    dimensions<k_order2> bidims2(index_range<k_order2>(idx1, idx2));

    // Loop over all se_label elements and do the reduction in each one
    std::set<std::string> id_done;
    for (; it1 != g1.end(); it1++) {

        const el1_t &se1 = g1.get_elem(it1);

#ifdef LIBTENSOR_DEBUG
        // Check that the block index dimensions are alright.
        if (bidims1 != se1.get_labeling().get_block_index_dims()) {
            throw bad_symmetry(g_ns, k_clazz, method, __FILE__, __LINE__,
                    "Incompatible se_labels in input.");
        }
#endif

        if (id_done.count(se1.get_table_id()) != 0) continue;

        // Combine all sym elements with the same product table
        combine_label<k_order1, T> cl1(se1);
        id_done.insert(cl1.get_table_id());
        typename adapter_t::iterator it2 = it1; it2++;
        for (; it2 != g1.end(); it2++) {
            const el1_t &se1b = g1.get_elem(it2);
            if (se1b.get_table_id() != cl1.get_table_id()) continue;

            cl1.add(se1b);
        }

        // Create result se_label
        el2_t se2(bidims2, cl1.get_table_id());

        // Transfer the labeling
        const block_labeling<k_order1> &bl1 = cl1.get_labeling();
        transfer_labeling(bl1, map, se2.get_labeling());

        // Obtain the product table
        const product_table_i &pt = product_table_container::get_instance().
                req_const_table(cl1.get_table_id());
        label_t n_labels = pt.get_n_labels();

        // Collect the labels for each reduction step!
        std::vector<label_group_t> blk_labels(nrsteps);
        for (size_t i = 0; i < nrsteps; i++) {

            // Find first dimensions of the current reduction step
            size_t j = 0;
            for (; j < k_order1; j++) {
                if (params.msk[j] && params.rseq[j] == i) break;
            }

            // Create a list of labels for the current reduction step
            size_t itype = bl1.get_dim_type(j);
            size_t imin = params.rblrange.get_begin()[j];
            size_t imax = params.rblrange.get_end()[j] + 1;

            label_group_t ibl(imax - imin, product_table_i::k_invalid);
            bool has_invalid = false;
            for (size_t k = imin, l = 0; k < imax; k++, l++) {
                ibl[l] = bl1.get_label(itype, k);
                if (ibl[l] == product_table_i::k_invalid) has_invalid = true;
            }

            // Check all other dimensions in the current reduction step.
            j++;
            for (; j < k_order1; j++) {
                if (! params.msk[j] || params.rseq[j] != i) continue;

                size_t jtype = bl1.get_dim_type(j);
                if (itype == jtype) continue;

                for (size_t k = imin, l = 0; k < imax; k++, l++) {
                    label_t ll = bl1.get_label(jtype, k);
                    if (ll == ibl[l]) continue;
                    if (ll == product_table_i::k_invalid) has_invalid = true;
                    else if (ibl[l] == product_table_i::k_invalid) ibl[l] = ll;
                    else {
#ifdef LIBTENSOR_DEBUG
                    throw bad_symmetry(g_ns, k_clazz, method, __FILE__,
                            __LINE__, "Incompatible block labels.");
#endif
                    }
                }
            }

            // Transfer the vector of block labels into a sorted vector of
            // unique labels
            label_group_t &cbl = blk_labels[i];
            if (has_invalid) {
                for (label_t ll = 0; ll < n_labels; ll++)
                    cbl.push_back(ll);
            }
            else {
                label_set_t ls;
                for (label_group_t::const_iterator il = ibl.begin();
                        il != ibl.end(); il++) {
                    ls.insert(*il);
                }
                for (label_set_t::const_iterator is = ls.begin();
                        is != ls.end(); is++) {
                    cbl.push_back(*is);
                }
            }
        }

        // Copy evaluation rule
        const evaluation_rule<N> &r1 = cl1.get_rule();
        evaluation_rule<k_order2> r2;

        // Map of sequences:
        // - if m1to2 == -1: no dimensions
        // - if m1to2 == r1.get_n_sequences(): only internal dimensions
        // - else sequence number in r2
        std::vector<size_t> m1to2(r1.get_n_sequences(), (size_t) -1);
        for (size_t sno = 0; sno < r1.get_n_sequences(); sno++) {

            const sequence<k_order1, size_t> &seq1 = r1[sno];
            sequence<k_order2, size_t> seq2(0);
            bool has_external = false, has_internal = false;
            for (register size_t i = 0; i < k_order1; i++) {
                if (seq1[i] == 0) continue;
                if (! params.msk[i]) {
                    has_external = true;
                    seq2[map[i]] = seq1[i];
                }
                else {
                    has_internal = true;
                }
            }

            if (has_external) m1to2[sno] = r2.add_sequence(seq2);
            else if (has_internal) m1to2[sno] = r1.get_n_sequences();
        }

        // Loop over the products
        size_t pno = 0;
        for (; pno < r1.get_n_products(); pno++) {

            // Loop over all terms in current product and find the reduction
            // indexes present

            // Reduction steps in product
            std::vector<bool> rsteps_in_pr(nrsteps, false);
            std::vector<bool> pr1to2_ni;
            std::vector<typename evaluation_rule<k_order1>::iterator> pr1to2;
            std::vector<typename evaluation_rule<k_order1>::iterator> pr1to0;

            bool product_forbidden = false, has_allowed = false;
            for (typename evaluation_rule<k_order1>::iterator itp =
                    r1.begin(pno); itp != r1.end(pno); itp++) {

                size_t sno = r1.get_seq_no(itp);
                // If the intrinsic label is the invalid label mark the product
                // as having an always allowed term and continue
                if (r1.get_intrinsic(itp) == product_table_i::k_invalid) {
                    has_allowed = true;
                    continue;
                }

                // If there the sequence is empty and not always allowed ...
                if (m1to2[sno] == (size_t) -1) {
                    // ... skip the product
                    product_forbidden = true;
                    break;
                }

                // Analyse how much internal dimensions there are
                const sequence<k_order1, size_t> &seq = r1.get_sequence(itp);
                size_t nk = 0;
                for (register size_t i = 0; i < k_order1; i++) {
                    if (! params.msk[i] || seq[i] == 0) continue;

                    rsteps_in_pr[params.rseq[i]] = true;
                    nk++;
                }

                // If there only internal dimensions add to pr1to0
                // else to pr1to2 / pr1to2_ni
                if (m1to2[sno] == r1.get_n_sequences()) {
                    if (nk > 0) pr1to0.push_back(itp);
                } else {
                    pr1to2.push_back(itp);
                    pr1to2_ni.push_back(nk == 0);
                }
            } // for itp

            // Skip the current product if one term is forbidden
            if (product_forbidden) continue;

            // If there are only always allowed terms in this product,
            // no further work has to be done
            if (pr1to2.size() == 0 && pr1to0.size() == 0 && has_allowed) {
                break;
            }

            // Create the dimensions for the reduction steps
            index<M> i1, i2;
            for (register size_t i = 0; i < nrsteps; i++) {
                if (! rsteps_in_pr[i]) continue;

                i2[i] = blk_labels[i].size() - 1;
            }
            dimensions<M> rdims(index_range<M>(i1, i2));

            // No loop over all possible index combinations of the reduction
            // dimensions and create possible combinations of result labels
            bool is_allowed = false;

            label_set_t tot_intr;
            abs_index<M> rx(rdims);
            do { // rx.inc()
                const index<M> &idx = rx.get_index();

                // Loop over terms w/o external indexes and check,
                // if they are allowed. If not skip the index
                size_t sno = 0;
                for (; sno < pr1to0.size(); sno++) {

                    typename evaluation_rule<k_order1>::iterator itp =
                            pr1to0[sno];
                    const sequence<k_order1, size_t> &seq =
                            r1.get_sequence(itp);

                    // Create a label group that comprises all internal block
                    // labels of the current index
                    label_group_t lg;
                    for (register size_t i = 0; i < k_order1; i++) {
                        if (seq[i] == 0) continue;

                        size_t k = params.rseq[i];
                        lg.insert(lg.end(), seq[i], blk_labels[k][idx[k]]);
                    }
                    lg.push_back(r1.get_intrinsic(itp));

                    // If the current term is forbidden stop
                    if (! pt.is_in_product(lg, r1.get_target(itp))) break;
                } // for sno

                // Skip this product if any of the terms is forbidden
                if (sno != pr1to0.size()) continue;

                // Stop here if current product contains only terms that are
                // always allowed
                if (pr1to2.size() == 0) {
                    is_allowed = true;
                    break;
                }

                // Loop over all other rules
                label_set_t cur_intr1, cur_intr2;
                cur_intr1.insert(product_table_i::k_identity);
                label_set_t *pi1 = &cur_intr1, *pi2 = &cur_intr2;
                label_t nlx = 1;
                for (size_t sno = 0; sno < pr1to2.size(); sno++) {

                    // Skip those w/o internal dims for now
                    if (pr1to2_ni[sno]) continue;

                    typename evaluation_rule<k_order1>::iterator itp =
                            pr1to2[sno];
                    const sequence<k_order1, size_t> &seq =
                            r1.get_sequence(itp);

                    // Create the label group of block labels of internal dims
                    // and intrinsic label
                    label_group_t lg;
                    for (register size_t i = 0; i < k_order1; i++) {
                        if (! params.msk[i] || seq[i] == 0) continue;

                        size_t k = params.rseq[i];
                        lg.insert(lg.end(), seq[i], blk_labels[k][idx[k]]);
                    }
                    lg.push_back(r1.get_intrinsic(itp));

                    // Compute the product and add all labels in the product
                    // to the combined label set tot_intr
                    label_set_t ls = pt.product(lg);
                    for (label_set_t::const_iterator ils1 = ls.begin();
                            ils1 != ls.end(); ils1++) {
                        label_t lx = *ils1 * nlx;
                        for (label_set_t::const_iterator ils2 = pi1->begin();
                                ils2 != pi1->end(); ils2++)
                            pi2->insert(*ils2 + lx);
                    }
                    std::swap(pi1, pi2);
                    pi2->clear();

                    nlx *= n_labels;
                } // for sno

                tot_intr.insert(pi1->begin(), pi1->end());
            } while (rx.inc());

            // If the product is unconditionally allowed stop here
            if (is_allowed) break;

            // Otherwise loop over all "total" intrinsic labels, decompose
            // each and set the respective product
            for (label_set_t::const_iterator ii = tot_intr.begin();
                    ii != tot_intr.end(); ii++) {

                label_t ll = *ii;
                size_t j = 0, ip;
                if (pr1to2_ni[j]) {
                    ip = r2.add_product(m1to2[r1.get_seq_no(pr1to2[j])],
                            r1.get_intrinsic(pr1to2[j]),
                            r1.get_target(pr1to2[j]));
                }
                else {
                    ip = r2.add_product(m1to2[r1.get_seq_no(pr1to2[j])],
                            ll % n_labels, r1.get_target(pr1to2[j]));
                    ll = ll / n_labels;
                }
                j++;

                for (; j < pr1to2.size(); j++) {
                    if (pr1to2_ni[j]) {
                        r2.add_to_product(ip, m1to2[r1.get_seq_no(pr1to2[j])],
                                r1.get_intrinsic(pr1to2[j]),
                                r1.get_target(pr1to2[j]));
                    }
                    else {
                        r2.add_to_product(ip, m1to2[r1.get_seq_no(pr1to2[j])],
                                ll % n_labels, r1.get_target(pr1to2[j]));
                        ll = ll / n_labels;
                    }
                } // for j
            } // for it
        } // for pno

        // This only happens if the result is an all allowed rule
        if (pno != r1.get_n_products()) {

            r2.clear_all();
            sequence<k_order2, size_t> seq(1);
            r2.add_sequence(seq);
            r2.add_product(0, product_table_i::k_invalid, 0);
        }

        // Return the product table
        product_table_container::get_instance().ret_table(cl1.get_table_id());

        se2.set_rule(r2);
        params.grp2.insert(se2);

    } // Loop it1
}


} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_LABEL_IMPL_H
