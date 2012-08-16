#ifndef LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H
#define LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H

#include <libtensor/core/permutation_builder.h>

namespace libtensor {


template<size_t N, size_t M, size_t NM, typename T>
const char *symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<NM, T> >::
k_clazz = "symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >";


template<size_t N, size_t M, size_t NM, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<NM, T> >::
do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(symmetry_operation_params_t&)";

    //  Adapter type for the input group
    typedef symmetry_element_set_adapter< N, T, se_perm<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_perm<M, T> > adapter2_t;

    mask<N + M> msk1, msk2;
    for (register size_t k = 0; k < N; k++) msk1[k] = true;
    for (register size_t k = N; k < N + M; k++) msk2[k] = true;

    // Simplest case: both source groups are empty!
    if (params.g1.is_empty() && params.g2.is_empty()) {
        params.g3.clear();
        return;
    }

    permutation_group<N + M, T> group;
    combine(params.g1, permutation<M>(), scalar_transf<T>(), group);
    combine(permutation<N>(), scalar_transf<T>(), params.g2, group);

    adapter1_t g1(params.g1);
    for(typename adapter1_t::iterator it = g1.begin();
            it != g1.end(); it++) {

        const se_perm<N, T> &e1 = g1.get_elem(it);
        if (e1.get_transf().is_identity()) continue;

        combine(e1.get_perm(), e1.get_transf(), params.g2, group);
    }
    adapter2_t g2(params.g2);
    for(typename adapter2_t::iterator it = g2.begin();
            it != g2.end(); it++) {

        const se_perm<M, T> &e2 = g2.get_elem(it);
        if (e2.get_transf().is_identity()) continue;

        combine(params.g1, e2.get_perm(), e2.get_transf(), group);
    }

    params.g3.clear();
    group.permute(params.perm);
    group.convert(params.g3);
}


template<size_t N, size_t M, size_t NM, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<NM, T> >::
combine(const permutation<N> &p1, const scalar_transf<T> &tr1,
        const symmetry_element_set<M, T> &set2,
        permutation_group<N + M, T> &grp) {

    typedef se_perm<M, T> se_perm_t;
    typedef symmetry_element_set_adapter<M, T, se_perm_t> adapter_t;
    typedef std::vector<se_perm_t> perm_vec_t;
    typedef std::list<se_perm_t> perm_lst_t;
    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;
    typedef std::list<transf_pair_t> tr_list_t;

    if (set2.is_empty()) return;

    // Separate set into 3 groups:
    // - elements with tr_e = tr
    // - elements with tr_e = 1
    // - all other elements
    adapter_t grp2(set2);
    perm_lst_t plst1, plst2;
    perm_vec_t plst3;
    tr_list_t tlst;
    for (typename adapter_t::iterator it = grp2.begin();
            it != grp2.end(); it++) {

        const se_perm_t &e2 = grp2.get_elem(it);
        if (e2.get_transf() == tr1) plst1.push_back(e2);
        else if (e2.get_transf().is_identity()) plst2.push_back(e2);
        else {
            tlst.push_back(transf_pair_t(plst3.size(), e2.get_transf()));
            plst3.push_back(e2);
        }
    }

    // Find all combinations of elements in the third group that yield
    // tr1 or 1 and add them to plst1 or plst2, respectively
    tr_list_t tlst2, *pt1 = &tlst, *pt2 = &tlst2;
    std::set<size_t> done;
    size_t nel = plst3.size(), n = 1;
    while (pt1->size() != 0) {
        for (typename tr_list_t::iterator it = pt1->begin();
                it != pt1->end(); it++) {

            size_t idxa = it->first * nel;
            for (size_t j = 0; j < nel; j++) {
                size_t idxb = idxa + j, k = 1, ix = idxb;
                while (idxb != 0) {
                    if (done.count(ix % nel)) break;
                    ix /= nel;
                }
                if (idxb != 0) continue;

                scalar_transf<T> &trx = it->second;
                trx.transform(plst3[j].get_transf());
                if (trx == tr1 || trx.is_identity()) {
                    done.insert(idxb);
                    std::vector<size_t> pidx;
                    pidx.reserve(n + 1);
                    while (idxb != 0) {
                        pidx.push_back(idxb % nel);
                        idxb /= nel;
                    }

                    permutation<M> py;
                    for (std::vector<size_t>::reverse_iterator ip =
                            pidx.rbegin(); ip != pidx.rend(); ip++) {
                        py.permute(plst3[*ip].get_perm());
                    }

                    if (trx == tr1)
                        plst1.push_back(se_perm_t(py, trx));
                    else
                        plst2.push_back(se_perm_t(py, trx));
                }
                else {
                    pt2->push_back(transf_pair_t(idxb, trx));
                }
            }
        }
        std::swap(pt1, pt2);
        pt2->clear();

        n++;
    }
    plst3.clear();
    done.clear();

    sequence<N + M, size_t> seq2a, seq2b;
    for (register size_t i = 0; i < N + M; i++) seq2a[i] = i;
    for (register size_t i = 0; i < N; i++) seq2b[i] = p1[i];

    // Combine all elements in p1 with all elements in p2
    if (! plst2.empty()) {
        perm_lst_t p1a;
        for (typename perm_lst_t::iterator it1 = plst2.begin();
                it1 != plst2.end(); it1++) {

            size_t orderp = it1->get_orderp();
            for (size_t i = 1; i < orderp; i++) {
                for (size_t j = 0; j < i; j++) {

                    for (typename perm_lst_t::iterator it2 = plst1.begin();
                            it2 != plst1.end(); it2++) {

                        permutation<M> px;
                        size_t k = 0;
                        for (; k < j; k++) px.permute(it1->get_perm());
                        px.permute(it2->get_perm());
                        for (; k < i; k++) px.permute(it1->get_perm());

                        for (register size_t i = 0, j = N; i < M; i++, j++)
                            seq2b[j] = px[i] + N;

                        permutation_builder<N + M> pb(seq2b, seq2a);
                        grp.add_orbit(tr1, pb.get_perm());
                    }
                }
            }
        }
    }
    else {

        for (typename perm_lst_t::iterator it = plst1.begin();
                it != plst1.end(); it++) {

            for (register size_t i = 0, j = N; i < M; i++, j++)
                seq2b[j] = it->get_perm()[i] + N;
            permutation_builder<N + M> pb(seq2b, seq2a);
            grp.add_orbit(tr1, pb.get_perm());
        }
    }
}


template<size_t N, size_t M, size_t NM, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<NM, T> >::
combine(const symmetry_element_set<N, T> &set1,
        const permutation<M> &p2, const scalar_transf<T> &tr2,
        permutation_group<N + M, T> &grp) {

    typedef se_perm<N, T> se_perm_t;
    typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;
    typedef std::vector<se_perm_t> perm_vec_t;
    typedef std::list<se_perm_t> perm_lst_t;
    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;
    typedef std::list<transf_pair_t> tr_list_t;

    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;

    if (set1.is_empty()) return;

    // Separate set into 3 groups:
    // - elements with tr_e = tr
    // - elements with tr_e = 1
    // - all other elements
    adapter_t grp1(set1);
    perm_lst_t plst1, plst2;
    perm_vec_t plst3;
    tr_list_t tlst;
    for (typename adapter_t::iterator it = grp1.begin();
            it != grp1.end(); it++) {

        const se_perm_t &e1 = grp1.get_elem(it);
        if (e1.get_transf() == tr2) plst1.push_back(e1);
        else if (e1.get_transf().is_identity()) plst2.push_back(e1);
        else {
            tlst.push_back(transf_pair_t(plst3.size(), e1.get_transf()));
            plst3.push_back(e1);
        }
    }

    // Find all combinations of elements in the third group that yield
    // tr1 or 1 and add them to plst1 or plst2, respectively
    tr_list_t tlst2, *pt1 = &tlst, *pt2 = &tlst2;
    std::set<size_t> done;
    size_t nel = plst3.size(), n = 1;
    while (pt1->size() != 0) {
        for (typename tr_list_t::iterator it = pt1->begin();
                it != pt1->end(); it++) {

            size_t idxa = it->first * nel;
            for (size_t j = 0; j < nel; j++) {
                size_t idxb = idxa + j, k = 1, ix = idxb;
                while (idxb != 0) {
                    if (done.count(ix % nel)) break;
                    ix /= nel;
                }
                if (idxb != 0) continue;

                scalar_transf<T> &trx = it->second;
                trx.transform(plst3[j].get_transf());
                if (trx == tr2 || trx.is_identity()) {
                    done.insert(idxb);
                    std::vector<size_t> pidx; pidx.reserve(n + 1);
                    while (idxb != 0) {
                        pidx.push_back(idxb % nel);
                        idxb /= nel;
                    }

                    permutation<N> py;
                    for (std::vector<size_t>::reverse_iterator ip =
                            pidx.rbegin(); ip != pidx.rend(); ip++) {
                        py.permute(plst3[*ip].get_perm());
                    }

                    if (trx == tr2)
                        plst1.push_back(se_perm_t(py, trx));
                    else
                        plst2.push_back(se_perm_t(py, trx));
                }
                else {
                    pt2->push_back(transf_pair_t(idxb, trx));
                }
            }
        }
        std::swap(pt1, pt2);
        pt2->clear();

        n++;
    }
    plst3.clear();
    done.clear();

    sequence<N + M, size_t> seq2a, seq2b;
    for (register size_t i = 0; i < N + M; i++) seq2a[i] = i;
    for (register size_t i = 0, j = N; i < M; i++, j++) seq2b[j] = p2[i] + N;

    // Combine all elements in p1 with all elements in p2
    if (! plst2.empty()) {
        perm_lst_t p1a;
        for (typename perm_lst_t::iterator it1 = plst2.begin();
                it1 != plst2.end(); it1++) {

            size_t orderp = it1->get_orderp();
            for (size_t i = 1; i < orderp; i++) {
                for (size_t j = 0; j < i; j++) {

                    for (typename perm_lst_t::iterator it2 = plst1.begin();
                            it2 != plst1.end(); it2++) {

                        permutation<N> px;
                        size_t k = 0;
                        for (; k < j; k++) px.permute(it1->get_perm());
                        px.permute(it2->get_perm());
                        for (; k < i; k++) px.permute(it1->get_perm());

                        for (register size_t i = 0; i < N; i++)
                            seq2b[i] = px[i];

                        permutation_builder<N + M> pb(seq2b, seq2a);
                        grp.add_orbit(tr2, pb.get_perm());
                    }
                }
            }
        }
    }
    else {

        for (typename perm_lst_t::iterator it = plst1.begin();
                it != plst1.end(); it++) {

            for (register size_t i = 0; i < N; i++)
                seq2b[i] = it->get_perm()[i];
            permutation_builder<N + M> pb(seq2b, seq2a);
            grp.add_orbit(tr2, pb.get_perm());
        }
    }
}


} // namespace libtensor


#endif // LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H
