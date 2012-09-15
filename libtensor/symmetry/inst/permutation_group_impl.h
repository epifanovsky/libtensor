#ifndef LIBTENSOR_PERMUTATION_GROUP_IMPL_H
#define LIBTENSOR_PERMUTATION_GROUP_IMPL_H

#include <algorithm>
#include <set>
#include <libtensor/core/permutation_builder.h>
#include <libtensor/defs.h>
#include "../bad_symmetry.h"

namespace libtensor {


template<size_t N, typename T>
const char *permutation_group<N, T>::k_clazz = "permutation_group<N, T>";


template<size_t N, typename T>
permutation_group<N, T>::permutation_group(
        const symmetry_element_set_adapter<N, T, se_perm_t> &set) {

    perm_list_t gs1, gs2;

    typedef symmetry_element_set_adapter<N, T, se_perm_t> adapter_t;
    //    std::cout << "ctor" << std::endl;
    for(typename adapter_t::iterator i = set.begin(); i != set.end(); i++) {
        //
        const se_perm_t &e = set.get_elem(i);
        //        std::cout << (e.is_symm() ? "symm" : "asymm") << " " << e.get_perm() << std::endl;
        gs1.push_back(gen_perm_t(e.get_perm(), e.get_transf()));
    }

    perm_list_t *p1 = &gs1, *p2 = &gs2;
    for(size_t i = 0; i < N; i++) {
        make_branching(m_br, i, *p1, *p2);
        std::swap(p1, p2);
        p2->clear();
    }
    //    std::cout << "Branching:" << std::endl;
    //    for (size_t i = 0; i < N; i++) {
    //        std::cout << i << " - " << m_br.m_edges[i] << ": ";
    //        std::cout << m_br.m_sigma[i].first <<
    //                "(" << (m_br.m_sigma[i].second ? '+' : '-') << ")" << ", " <<
    //                m_br.m_tau[i].first <<
    //                "(" << (m_br.m_sigma[i].second ? '+' : '-') << ")" << std::endl;
    //    }
    //
    //    std::cout << "ctor end" << std::endl;
}


template<size_t N, typename T>
void permutation_group<N, T>::add_orbit(const scalar_transf<T> &tr,
        const permutation<N> &perm) {

    static const char *method =
            "add_orbit(const scalar_transf<T>&, const permutation<N>&)";

    if (perm.is_identity()) {
        if (tr.is_identity()) return;
        throw bad_symmetry(g_ns, k_clazz, method,
                __FILE__, __LINE__, "perm");
    }

    scalar_transf<T> trx(tr);
    if (is_member(m_br, 0, trx, perm)) {
        if (trx.is_identity()) return;
        throw bad_symmetry(g_ns, k_clazz, method,
                __FILE__, __LINE__, "tr");
    }

    perm_list_t gs1, gs2;
    make_genset(m_br, gs1);
    gs1.push_back(gen_perm_t(perm, tr));
    m_br.reset();

    perm_list_t *p1 = &gs1, *p2 = &gs2;
    for(size_t i = 0; i < N; i++) {
        make_branching(m_br, i, *p1, *p2);
        std::swap(p1, p2);
        p2->clear();
    }
}


template<size_t N, typename T>
void permutation_group<N, T>::convert(symmetry_element_set<N, T> &set) const {

    perm_list_t gs;

    make_genset(m_br, gs);
    for(typename perm_list_t::iterator i = gs.begin(); i != gs.end(); i++) {
        set.insert(se_perm_t(i->first, i->second));
    }
    gs.clear();
}


template<size_t N, typename T> template<size_t M>
void permutation_group<N, T>::project_down(const mask<N> &msk,
        permutation_group<M, T> &g2) {

    static const char *method =
            "project_down<M>(const mask<N>&, permutation_group<M, T>&)";

    register size_t m = 0;
    for(register size_t i = 0; i < N; i++) if(msk[i]) m++;
    if(m != M) {
        throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__, "msk");
    }

    branching br;
    perm_list_t gs1, gs2;
    perm_list_t *p1 = &gs1, *p2 = &gs2;
    make_genset(m_br, gs1);

    for(size_t i = 0; i < N; i++) {
        if(msk[i]) continue;
        //      std::cout << "genset before branching of " << i << ": <";
        //      for (typename perm_list_t::const_iterator pi = p1->begin();
        //              pi != p1->end(); pi++)
        //          std::cout << " " << *pi;
        //      std::cout << ">" << std::endl;
        br.reset();
        make_branching(br, i, *p1, *p2);
        std::swap(p1, p2);
        p2->clear();
    }
    //  std::cout << "genset1: <";
    //  for(typename perm_list_t::const_iterator pi = p1->begin();
    //      pi != p1->end(); pi++) {
    //      std::cout << " " << *pi;
    //  }
    //  std::cout << " >" << std::endl;
    //  std::cout << "genset2: <";
    for(typename perm_list_t::const_iterator pi = p1->begin();
            pi != p1->end(); pi++) {

        sequence<N, size_t> seq1a(0), seq2a(0);
        sequence<M, size_t> seq1b(0), seq2b(0);
        for(size_t i = 0; i < N; i++) seq2a[i] = seq1a[i] = i;
        pi->first.apply(seq2a);
        size_t j = 0;
        for(size_t i = 0; i < N; i++) {
            if(!msk[i]) continue;
            seq1b[j] = seq1a[i];
            seq2b[j] = seq2a[i];
            j++;
        }
        permutation_builder<M> pb(seq2b, seq1b);
        //      std::cout << " " << pb.get_perm();
        g2.add_orbit(pi->second, pb.get_perm());
    }
    //  std::cout << " >" << std::endl;
}


template<size_t N, typename T>
void permutation_group<N, T>::stabilize(
        const mask<N> &msk, permutation_group<N, T> &g2) {

    sequence<N, size_t> seq(0);
    for (register size_t i = 0; i != N; i++) {
        if (msk[i]) seq[i] = 1;
    }
    stabilize(seq, g2);
}


template<size_t N, typename T>
void permutation_group<N, T>::stabilize(
        const sequence<N, size_t> &seq, permutation_group<N, T> &g2) {

    // generating set of G(P)
    perm_list_t gs;
    make_setstabilizer(m_br, seq, gs);
    for (typename perm_list_t::const_iterator pi = gs.begin();
            pi != gs.end(); pi++) {
        g2.add_orbit(pi->second, pi->first);
    }
    gs.clear();
}


template<size_t N, typename T>
void permutation_group<N, T>::permute(const permutation<N> &perm) {

    if(perm.is_identity()) return;
    permute_branching(m_br, perm);
}


template<size_t N, typename T>
size_t permutation_group<N, T>::get_path(
        const branching &br, size_t i, size_t j, size_t (&path)[N]) const {

    if(j <= i) return 0;

    size_t p[N];
    register size_t k = j;
    register size_t len = 0;
    while(k != N && k != i) {
        p[len++] = k;
        k = br.m_edges[k];
    }
    if(k != i) return 0;

    for(k = 0; k < len; k++) {
        path[k] = p[len - k - 1];
    }
    return len;
}


template<size_t N, typename T>
bool permutation_group<N, T>::is_member(const branching &br, size_t i,
        scalar_transf<T> &tr, const permutation<N> &perm) const {

    //  Check the identity first
    //
    if(perm.is_identity()) return true;
    if(i >= N - 1) return false;

    //  Find the element pi1 of the right coset representative Ui
    //  for which rho = pi * pi1^{-1} stabilizes i. (pi == perm).

    if(perm[i] == i) {
        return is_member(br, i + 1, tr, perm);
    }

    //  Go over non-identity members of Ui
    //
    for(size_t j = i + 1; j < N; j++) {

        size_t path[N];
        size_t pathlen = get_path(br, i, j, path);
        if(pathlen == 0) continue;

        // sij = ti^-1 tj
        // rho = p sij^-1  = p tj^-1 ti;
        perm_t rho(br.m_tau[i].first), tjinv(br.m_tau[j].first, true);
        rho.permute(tjinv).permute(perm);

        transf_t tr_rho(br.m_tau[i].second), tr_tjinv(br.m_tau[j].second);
        tr_tjinv.invert();
        tr_rho.transform(tr_tjinv).transform(tr);

        if(rho[i] == i) {
            if (is_member(br, i + 1, tr_rho, rho)) {
                tr = tr_rho;
                return true;
            }
        }
    }
    return false;
}


template<size_t N, typename T>
void permutation_group<N, T>::make_branching(branching &br, size_t i,
        const perm_list_t &gs, perm_list_t &gs2) {

    if(gs.empty()) return;

    perm_vec_t transv(N);
//    std::cout << "make_branching" << std::endl;
//    std::cout << "transversal(" << i << ")" << std::endl;
//    std::cout << "genset: <";
//    for(typename perm_list_t::const_iterator pi = gs.begin();
//            pi != gs.end(); pi++) {
//        std::cout << " " << pi->first << "(" << pi->second << ")";
//    }
//    std::cout << " >" << std::endl;

    std::vector<size_t> delta;
    delta.push_back(i);

    std::list<size_t> s;
    s.push_back(i);

    transv[i].first.reset();
    transv[i].second.reset();

    while(! s.empty()) {

        size_t j = s.front();
        s.pop_front();

        for(typename perm_list_t::const_iterator pi = gs.begin();
                pi != gs.end(); pi++) {

            sequence<N, size_t> seq(0);
            for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
            pi->first.apply(seq);

            size_t k = seq[j];
            typename std::vector<size_t>::iterator dd = delta.begin();
            while(dd != delta.end() && *dd != k) dd++;
            if(dd == delta.end()) {
                gen_perm_t p(*pi);
                p.first.permute(transv[j].first);
                p.second.transform(transv[j].second);
                transv[k].first.reset();
                transv[k].first.permute(p.first);
                transv[k].second.reset();
                transv[k].second.transform(p.second);
                delta.push_back(k);
                s.push_back(k);
            }
        }
    }

//    std::cout << "transv: {";
//    for(size_t j = 0; j < N; j++) {
//        std::cout << " " << transv[j].first <<
//                "(" << transv[j].second << ")";
//    }
//    std::cout << " }" << std::endl;

    for(typename std::vector<size_t>::iterator dd = delta.begin();
            dd != delta.end(); dd++) {

        size_t j = *dd;
        if(j == i) continue;

        // add a new edge (remove an existing edge if necessary)
        br.m_edges[j] = i;
        br.m_sigma[j].first.reset();
        br.m_sigma[j].first.permute(transv[j].first);
        br.m_sigma[j].second.reset();
        br.m_sigma[j].second.transform(transv[j].second);
        br.m_tau[j].first.reset();
        br.m_tau[j].first.permute(br.m_sigma[j].first);
        br.m_tau[j].first.permute(br.m_tau[i].first);
        br.m_tau[j].second.reset();
        br.m_tau[j].second.transform(br.m_sigma[j].second);
        br.m_tau[j].second.transform(br.m_tau[i].second);
    }

//        std::cout << "graph: {" << std::endl;
//        for(size_t j = 0; j < N; j++) {
//            size_t k = br.m_edges[j];
//            if(k == N) continue;
//            permutation<N> pinv(br.m_sigma[j].first, true);
//            std::cout << j << "->" << k <<
//                    " " << br.m_sigma[j].first <<
//                    "(" << br.m_sigma[j].second << ")" <<
//                    " " << br.m_tau[j].first <<
//                    "(" << br.m_tau[j].second << ")" <<
//                    " " << k << "->" << j << " " << pinv << std::endl;
//        }
//        std::cout << "}" << std::endl;

    for(typename perm_list_t::const_iterator pi = gs.begin();
            pi != gs.end(); pi++) {

        sequence<N, size_t> seq(0);
        for(size_t ii = 0; ii < N; ii++) seq[ii] = ii;
        pi->first.apply(seq);

        for(typename std::vector<size_t>::iterator dd = delta.begin();
                dd != delta.end(); dd++) {

            size_t j = *dd, k = seq[j];
            gen_perm_t p(transv[k]);
            p.first.invert();
            p.first.permute(pi->first).permute(transv[j].first);
            p.second.invert();
            p.second.transform(pi->second).transform(transv[j].second);
            if(! p.first.is_identity()) {
                typename perm_list_t::const_iterator rho = gs2.begin();
                while(rho != gs2.end() && ! rho->first.equals(p.first)) rho++;
                if(rho == gs2.end()) gs2.push_back(p);
            }
            else if (! p.second.is_identity()) {
                throw generic_exception(g_ns, k_clazz,
                        "make_branching(branching, size_t, "
                        "const perm_list_t &, perm_list_t&)",
                        __FILE__, __LINE__, "Illegal permutation.");
            }
        }
    }

    //    std::cout << "genset2: <";
    //    for(typename perm_list_t::const_iterator pi = gs2.begin();
    //            pi != gs2.end(); pi++) {
    //
    //        std::cout << " " << pi->first << "(" << (pi->second ? '+' : '-') << ")";
    //    }
    //    std::cout << " >" << std::endl;

}


template<size_t N, typename T>
void permutation_group<N, T>::make_genset(
        const branching &br, perm_list_t &gs) const {

    for(register size_t i = 0; i < N; i++) {
        if(br.m_edges[i] != N && ! br.m_sigma[i].first.is_identity()) {
            gs.push_back(br.m_sigma[i]);
        }
    }
    //~ std::cout << "genset: <";
    //~ for(typename perm_list_t::const_iterator pi = gs.begin();
    //~ pi != gs.end(); pi++) {

    //~ std::cout << " " << *pi;
    //~ }
    //~ std::cout << " >" << std::endl;
}


template<size_t N, typename T>
void permutation_group<N, T>::permute_branching(
        branching &br, const permutation<N> &perm) {

    //~ std::cout << "graph(bef): {" << std::endl;
    //~ for(size_t j = 0; j < N; j++) {
    //~ size_t k = br.m_edges[j];
    //~ if(k == N) continue;
    //~ permutation<N> pinv(br.m_sigma[j], true);
    //~ std::cout << k << "->" << j << " " << br.m_sigma[j] << " " << br.m_tau[j]
    //~ << " " << j << "->" << k << " " << pinv << std::endl;
    //~ }
    //~ std::cout << "}" << std::endl;
    //~ std::cout << "perm: " << perm << std::endl;

    perm_list_t gs1, gs2, gs3;
    make_genset(br, gs1);
    for(typename perm_list_t::iterator i = gs1.begin();
            i != gs1.end(); i++) {

        sequence<N, size_t> seq1(0), seq2(0);
        for(size_t j = 0; j < N; j++) seq2[j] = seq1[j] = j;
        i->first.apply(seq2);
        permutation_builder<N> pb(seq2, seq1, perm);
        gs2.push_back(gen_perm_t(pb.get_perm(), i->second));
    }
    br.reset();
    perm_list_t *p1 = &gs2, *p2 = &gs3;
    for(size_t i = 0; i < N; i++) {
        make_branching(br, i, *p1, *p2);
        std::swap(p1, p2);
        p2->clear();
    }

    //~ std::cout << "graph(aft): {" << std::endl;
    //~ for(size_t j = 0; j < N; j++) {
    //~ size_t k = br.m_edges[j];
    //~ if(k == N) continue;
    //~ permutation<N> pinv(br.m_sigma[j], true);
    //~ std::cout << k << "->" << j << " " << br.m_sigma[j] << " " << br.m_tau[j]
    //~ << " " << j << "->" << k << " " << pinv << std::endl;
    //~ }
    //~ std::cout << "}" << std::endl;
}

template<size_t N, typename T>
void permutation_group<N, T>::make_setstabilizer(const branching &br,
        const sequence<N, size_t> &msk, perm_list_t &gs) {

    static const char *method = "make_set_stabilizer(const branching &, "
            "const mask<N>&, perm_list_t&)";

    register size_t m = 0;
    for(register size_t i = 0; i < N; i++) if(msk[i] != 0) m++;
    // Handle two special cases first:
    // 1) mask is empty -> no stabilization
    if(m == 0) {
        make_genset(br, gs);
        return;
    }
    // 2) one index is masked -> point stabilization
    if(m == 1) {
        branching brx;
        perm_list_t gsx;
        make_genset(br, gsx);
        size_t i = 0;
        for(; i < N; i++) if (msk[i] != 0) break;
        make_branching(brx, i, gsx, gs);
        return;
    }

    // Backtrack search for group elements that fulfill the set stabilizing
    // criterion P. These elements form a subgroup G(P).

    // The algorithm is as follows:
    // Loop over all stabilizer subgroups starting from G_N = <()> to G_0
    // In each step compute (G_i \ G_{i+1}) \cap G(P) and add to gs

    // Loop over all stabilizers G_i starting from G_{N-1} since G_N = <()>
    for (size_t i = N - 1, ii = i - 1; i > 0; i--, ii--) {
        // ii is the proper index!!!

//        std::cout << ii << std::endl;

        // Compute all elements of G_i \cap G(P)

        // we need N - ii permutations to build permutation
        // g = u_{N - 1} ... u_{ii}
        perm_vec_t pu(N - i); // vector u with u_i \in U_i
        std::vector<size_t> ui(N - i);
        // initialize ui and pu
        for (size_t k = ii; k < N - 1; k++) ui[k - ii] = k;
        ui[0]++;

        for (; ui[0] < N; ui[0]++) {
            size_t k = br.m_edges[ui[0]];
            while (k != ii && k != N) k = br.m_edges[k];
            if (k == N) continue;
            pu[0].first.reset();
            pu[0].first.permute(br.m_tau[ui[0]].first);
            pu[0].first.permute(perm_t(br.m_tau[k].first, true));
            transf_t tr_tkinv(br.m_tau[k].second); tr_tkinv.invert();
            pu[0].second.reset();
            pu[0].second.transform(br.m_tau[ui[0]].second);
            pu[0].second.transform(tr_tkinv);
            break;
        }

        // loop over all possible sequences u_{N-1} ... u_{ii}
        while (ui[0] != N) {

            gen_perm_t g;

            // build the permutation g = u_{N-1} ... u_{ii}
            for (size_t k = 0; k < ui.size(); k++) {
                if (ui[k] == k + ii) continue;
                g.first.permute(pu[k].first);
                g.second.transform(pu[k].second);
            }

            // check whether g is in G(P)
            sequence<N, size_t> seq(0);
            for (size_t k = 0; k < N; k++) seq[k] = k;
            g.first.apply(seq);

//            std::cout << "g: [";
//            for (size_t k = 0; k < N; k++) std::cout << seq[k];
//            std::cout << "]" << std::endl;

            // Check if g stabilizes the set
            mask<N> done;
            size_t l = 0;
            for (; l < N; l++) {
                if (done[l]) continue;
                if (msk[seq[l]] == 0) {
                    if (msk[l] != 0) break;
                    else continue;
                }

                size_t m = l + 1;
                for (; m < N; m++) {
                    if (msk[m] != msk[l]) continue;
                    done[m] = true;
                    if (msk[seq[m]] != msk[seq[l]]) break;
                }
                if (m != N) break;
            }

            // If g is in G(P), we add it to the list of permutations
            if (l == N) gs.push_back(g);

            // Now, go to the next sequence u_{N-1} ... u_ii
            for (size_t k = ui.size(), k1 = k - 1; k > 0; k--, k1--) {
                ui[k1]++;
                for (; ui[k1] < N; ui[k1]++) {
                    size_t m = br.m_edges[ui[k1]];
                    while (m != ii + k1 && m != N) m = br.m_edges[m];
                    if (m == N) continue;
                    pu[k1].first.reset();
                    pu[k1].first.permute(br.m_tau[ui[k1]].first);
                    pu[k1].first.permute(perm_t(br.m_tau[m].first, true));
                    transf_t tr_tminv(br.m_tau[m].second); tr_tminv.invert();
                    pu[k1].second.reset();
                    pu[k1].second.transform(br.m_tau[ui[k1]].second);
                    pu[k1].second.transform(tr_tminv);
                    break;
                }
                if (ui[k1] != N || k1 == 0) break;
                ui[k1] = ii + k1;
                pu[k1].first.reset();
                pu[k1].second.reset();
            }
        }
    }
}

} // namespace libtensor

#endif // LIBTENSOR_PERMUTATION_GROUP_IMPL_H
