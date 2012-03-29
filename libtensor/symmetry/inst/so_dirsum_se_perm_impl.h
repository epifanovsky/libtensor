#ifndef LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H
#define LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H

#include <libtensor/core/permutation_builder.h>

namespace libtensor {


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >::
k_clazz = "symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >::
do_perform(symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(symmetry_operation_params_t&)";

    //	Adapter type for the input group
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


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >::
combine(const permutation<N> &p1, const scalar_transf<T> &tr1,
        const symmetry_element_set<M, T> &set2,
        permutation_group<N + M, T> &grp) {

    typedef se_perm<M, T> se_perm_t;
    typedef std::vector<se_perm_t> perm_vec_t;
    typedef std::list<se_perm_t> perm_lst_t;
    typedef symmetry_element_set_adapter<M, T, se_perm_t> adapter_t;

    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;

    if (set2.is_empty()) return;

    // Separate set into 3 groups:
    // - elements with tr_e = tr
    // - elements with tr_e = 1
    // - all other elements
    adapter_t grp2(set2);
    perm_lst_t pl1, pl2;
    perm_vec_t pl3;
    for (typename adapter_t::iterator it = grp2.begin();
            it != grp2.end(); it++) {

        const se_perm_t &e2 = grp2.get_elem(it);
        if (e2.get_transf() == tr1) pl1.push_back(e2);
        else if (e2.get_transf().is_identity()) pl2.push_back(e2);
        else pl3.push_back(e2);
    }

    // Find all combinations of elements in the third group that yield
    // tr1 and add them to pl1
    if (! pl3.empty()) {

        size_t nel = pl3.size();
        std::vector<size_t> order(nel, 1);
        std::vector<transf_pair_t> s1, s2, *ps1 = &s1, *ps2 = &s2;
        std::set<size_t> done;
        for (size_t i = 0; i < nel; i++) {
            scalar_transf<T> trx(pl3[i].get_transf());
            while (! trx.is_identity()) {
                trx.transform(pl3[i].get_transf());
                order[i]++;
            }
            s1.push_back(transf_pair_t(i, pl3[i].get_transf()));
        }

        size_t max = lcm(order), nf = 1;
        for (size_t n = 1; n <= max; n++, nf *= nel) {
            for (size_t i = 0; i < ps1->size(); i++) {
                size_t idxa = (*ps1)[i].first * nf;
                for (size_t j = 0; j < nel; j++) {
                    size_t idxb = idxa + j, k = 1, ix = idxb;
                    for (; k < n; k++) {
                        if (done.count(ix % nel)) break;
                        ix /= nel;
                    }
                    if (k != n) continue;

                    scalar_transf<T> &trx = (*ps1)[i].second;
                    trx.transform(pl3[j].get_transf());
                    if (trx == tr1) {
                        done.insert(idxb);
                        permutation<M> py;
                        while (idxb != 0) {
                            py.permute(pl3[idxb % nel].get_perm());
                            idxb /= nel;
                        }
                        py.invert();
                        pl1.push_back(se_perm_t(py, trx));
                    }
                    else {
                        ps2->push_back(transf_pair_t(idxb, trx));
                    }
                }
                std::swap(ps1, ps2);
                ps2->clear();

                if (ps1->size() == 0) break;
            }
        }
        pl3.clear();
    }

    sequence<N + M, size_t> seq2a, seq2b;
    for (register size_t i = 0; i < N + M; i++) seq2a[i] = i;
    for (register size_t i = 0; i < N; i++) seq2b[i] = p1[i];

    // Combine all elements in p1 with all elements in p2
    if (! pl2.empty()) {
        perm_lst_t p1a;
        for (typename perm_lst_t::iterator it1 = pl2.begin();
                it1 != pl2.end(); it1++) {

            size_t orderp = it1->get_orderp();
            for (size_t i = 1; i < orderp; i++) {
                for (size_t j = 0; j < i; j++) {

                    for (typename perm_lst_t::iterator it2 = pl1.begin();
                            it2 != pl1.end(); it2++) {

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

        for (typename perm_lst_t::iterator it = pl1.begin();
                it != pl1.end(); it++) {

            for (register size_t i = 0, j = N; i < M; i++, j++)
                seq2b[j] = it->get_perm()[i] + N;
            permutation_builder<N + M> pb(seq2b, seq2a);
            grp.add_orbit(tr1, pb.get_perm());
        }
    }
}


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >::
combine(const symmetry_element_set<N, T> &set1,
        const permutation<M> &p2, const scalar_transf<T> &tr2,
        permutation_group<N + M, T> &grp) {

    typedef se_perm<N, T> se_perm_t;
    typedef std::vector<se_perm_t> perm_vec_t;
    typedef std::list<se_perm_t> perm_lst_t;
    typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;

    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;

    if (set1.is_empty()) return;

    // Separate set into 3 groups:
    // - elements with tr_e = tr
    // - elements with tr_e = 1
    // - all other elements
    adapter_t grp1(set1);
    perm_lst_t pl1, pl2;
    perm_vec_t pl3;
    for (typename adapter_t::iterator it = grp1.begin();
            it != grp1.end(); it++) {

        const se_perm_t &e1 = grp1.get_elem(it);
        if (e1.get_transf() == tr2) pl1.push_back(e1);
        else if (e1.get_transf().is_identity()) pl2.push_back(e1);
        else pl3.push_back(e1);
    }

    // Find all combinations of elements in the third group that yield
    // tr_e = tr and add them to p1
    if (! pl3.empty()) {

        size_t nel = pl3.size();
        std::vector<size_t> order(nel, 1);
        std::vector<transf_pair_t> s1, s2, *ps1 = &s1, *ps2 = &s2;
        std::set<size_t> done;
        for (size_t i = 0; i < nel; i++) {
            scalar_transf<T> trx(pl3[i].get_transf());
            while (! trx.is_identity()) {
                trx.transform(pl3[i].get_transf());
                order[i]++;
            }
            s1.push_back(transf_pair_t(i, pl3[i].get_transf()));
        }

        size_t max = lcm(order), nf = 1;
        for (size_t n = 1; n <= max; n++, nf *= nel) {
            for (size_t i = 0; i < ps1->size(); i++) {
                size_t idxa = (*ps1)[i].first * nf;
                for (size_t j = 0; j < nel; j++) {
                    size_t idxb = idxa + j, k = 1, ix = idxb;
                    for (; k < n; k++) {
                        if (done.count(ix % nel)) break;
                        ix /= nel;
                    }
                    if (k != n) continue;

                    scalar_transf<T> &trx = (*ps1)[i].second;
                    trx.transform(pl3[j].get_transf());
                    if (trx == tr2) {
                        done.insert(idxb);
                        permutation<N> py;
                        while (idxb != 0) {
                            py.permute(pl3[idxb % nel].get_perm());
                            idxb /= nel;
                        }
                        py.invert();
                        pl1.push_back(se_perm_t(py, trx));
                    }
                    else {
                        ps2->push_back(transf_pair_t(idxb, trx));
                    }
                }
                std::swap(ps1, ps2);
                ps2->clear();

                if (ps1->size() == 0) break;
            }
        }
        pl3.clear();
    }

    sequence<N + M, size_t> seq2a, seq2b;
    for (register size_t i = 0; i < N + M; i++) seq2a[i] = i;
    for (register size_t i = 0, j = N; i < M; i++, j++) seq2b[j] = p2[i] + N;

    // Combine all elements in p1 with all elements in p2
    if (! pl2.empty()) {
        perm_lst_t p1a;
        for (typename perm_lst_t::iterator it1 = pl2.begin();
                it1 != pl2.end(); it1++) {

            size_t orderp = it1->get_orderp();
            for (size_t i = 1; i < orderp; i++) {
                for (size_t j = 0; j < i; j++) {

                    for (typename perm_lst_t::iterator it2 = pl1.begin();
                            it2 != pl1.end(); it2++) {

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

        for (typename perm_lst_t::iterator it = pl1.begin();
                it != pl1.end(); it++) {

            for (register size_t i = 0; i < N; i++)
                seq2b[i] = it->get_perm()[i];
            permutation_builder<N + M> pb(seq2b, seq2a);
            grp.add_orbit(tr2, pb.get_perm());
        }
    }
}

template<size_t N, size_t M, typename T>
size_t symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<N + M, T> >::lcm(
        const std::vector<size_t> &seq) {

    if (seq.size() == 1) return seq[0];

#ifdef LIBTENSOR_DEBUG
    for (register size_t i = 0; i < N; i++) {
        if (seq[i] == 0) {
            throw bad_parameter(g_ns, k_clazz,
                    "lcm(const std::vector<size_t>&)",
                    __FILE__, __LINE__, "seq");
        }
    }
#endif

    std::vector<size_t> seq2(seq);
    do {
        register size_t i = 1;
        for (; i < seq2.size() && seq2[i] == seq2[0]; i++) ;
        if (i == seq2.size()) break;

        size_t imin = 0;
        for (i = 1; i < seq2.size(); i++) {
            if (seq2[i] < seq2[imin]) imin = i;
        }
        seq2[imin] += seq[i];
    } while (true);

    return seq2[0];
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_PERM_IMPL_H
