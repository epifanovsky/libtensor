#ifndef LIBTENSOR_SO_APPLY_SE_PERM_IMPL_H
#define LIBTENSOR_SO_APPLY_SE_PERM_IMPL_H

#include "../permutation_group.h"


namespace libtensor {


template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::k_clazz =
        "symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_perm<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    //	Adapter type for the input group
    typedef symmetry_element_set_adapter<N, T, element_t> adapter_t;
    typedef std::vector<element_t> perm_vec_t;
    typedef std::pair< size_t, scalar_transf<T> > transf_pair_t;
    typedef std::list<transf_pair_t> transf_list_t;

    scalar_transf<T> tr(params.s2);
    for (register size_t i = 1; i < N && ! tr.is_identity(); i++) {
        tr.transform(params.s2);
    }
    bool is_cyclic = tr.is_identity();

    params.grp2.clear();

    adapter_t adapter(params.grp1);
    permutation_group<N, T> grp2;

    perm_vec_t plst;
    transf_list_t tlst;
    for (typename adapter_t::iterator it = adapter.begin();
            it != adapter.end(); it++) {

        const element_t &el = adapter.get_elem(it);
        if (el.get_transf().is_identity()) {
            grp2.add_orbit(el.get_transf(), el.get_perm());
        }
        else if (is_cyclic) {
            if (el.get_transf() == params.s1) {
                grp2.add_orbit(params.s2, el.get_perm());
            }
            else {
                tlst.push_back(transf_pair_t(plst.size(), el.get_transf()));
                plst.push_back(el);
            }
        }
    }

    size_t nel = plst.size(), n = 1;
    transf_list_t tlst2, *pt1 = &tlst, *pt2 = &tlst2;
    std::set<size_t> done;
    while (pt1->size() != 0) {
        for (typename transf_list_t::iterator it = pt1->begin();
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
                trx.transform(plst[j].get_transf());
                if (trx == params.s1 || trx.is_identity()) {
                    done.insert(idxb);
                    std::vector<size_t> pidx; pidx.reserve(n + 1);
                    while (idxb != 0) {
                        pidx.push_back(idxb % nel);
                        idxb /= nel;
                    }

                    permutation<N> py;
                    for (std::vector<size_t>::reverse_iterator ip =
                            pidx.rbegin(); ip != pidx.rend(); ip++) {
                        py.permute(plst[*ip].get_perm());
                    }

                    if (trx.is_identity())
                        grp2.add_orbit(trx, py);
                    else
                        grp2.add_orbit(params.s2, py);
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

    grp2.permute(params.perm1);
    grp2.convert(params.grp2);
}


} // namespace libtensor


#endif // LIBTENSOR_SO_APPLY_IMPL_PERM_H
