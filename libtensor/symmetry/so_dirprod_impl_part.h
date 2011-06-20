#ifndef LIBTENSOR_SO_DIRPROD_IMPL_PART_H
#define LIBTENSOR_SO_DIRPROD_IMPL_PART_H

#include <map>
#include "../defs.h"
#include "../exception.h"
#include "../not_implemented.h"
#include "../core/permutation_builder.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "partition_set.h"
#include "so_dirprod.h"
#include "se_part.h"

namespace libtensor {


/**	\brief Implementation of so_dirprod<N, M, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_dirprod<N, M, T>, se_part<N + M, T> > :
public symmetry_operation_impl_base< so_dirprod<N, M, T>, se_part<N + M, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirprod<N, M, T> operation_t;
    typedef se_part<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};


template<size_t N, size_t M, typename T>
const char *symmetry_operation_impl< so_dirprod<N, M, T>,
se_part<N + M, T> >::k_clazz =
        "symmetry_operation_impl< so_dirprod<N, M, T>, se_part<N, T> >";


template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_dirprod<N, M, T>,
se_part<N + M, T> >::do_perform(symmetry_operation_params_t &params) const {

    static const char *method = "do_perform(symmetry_operation_params_t&)";

    // Adapter type for the input groups
    typedef symmetry_element_set_adapter< N, T, se_part<N, T> > adapter1_t;
    typedef symmetry_element_set_adapter< M, T, se_part<M, T> > adapter2_t;

#ifdef LIBTENSOR_DEBUG
    //    consistency< se_part<N, T> >(params.g1).check();
    //    consistency< se_part<M, T> >(params.g2).check();
#endif

    adapter1_t g1(params.g1);
    adapter2_t g2(params.g2);
    params.g3.clear();

    // map result index to input index
    sequence<N + M, size_t> map(0);
    for (size_t i = 0; i < N + M; i++) map[i] = i;
    params.perm.apply(map);

    mask<N + M> msk1, msk2;
    sequence<N, size_t> seq1a(0), seq1b(0);
    sequence<M, size_t> seq2a(0), seq2b(0);
    for (size_t i = 0, j = 0, k = 0; i < N + M; i++) {
        if (map[i] < N) {
            msk1[i] = true;
            seq1a[j] = j; seq1b[j] = map[i];
            j++;
        }
        else {
            msk2[i] = true;
            seq2a[k] = k; seq2b[k] = map[i] - N;
            k++;
        }
    }

    permutation_builder<N> pb1(seq1b, seq1a);
    permutation_builder<M> pb2(seq2b, seq2a);
    permutation<N> p1inv(pb1.get_perm(), true);
    permutation<M> p2inv(pb2.get_perm(), true);

    // Determine total masks for every number of partitions
    typedef typename std::map< size_t, mask<N + M> > map_t;
    map_t msks;
    for(typename adapter1_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        const se_part<N, T> &e = g1.get_elem(it1);
        const mask<N> &mx = e.get_mask();
        mask<N + M> m;
        for (size_t i = 0, j = 0; i < N + M; i++)
            if (msk1[i]) { m[i] = mx[j]; j++; }
        msks[e.get_npart()] |= m;
    }
    for(typename adapter2_t::iterator it2 = g2.begin();
            it2 != g2.end(); it2++) {

        const se_part<M, T> &e = g2.get_elem(it2);
        const mask<M> &mx = e.get_mask();
        mask<N + M> m;
        for (size_t i = 0, j = 0; i < N + M; i++)
            if (msk2[i]) { m[i] = mx[j]; j++; }
        msks[e.get_npart()] |= m;
    }

    // Now loop over all npart
    for(typename map_t::const_iterator it = msks.begin();
            it != msks.end(); it++) {

        se_part<N + M, T> part(params.bis, it->second, it->first);

        for(typename adapter1_t::iterator it1 = g1.begin();
                it1 != g1.end(); it1++) {

            const se_part<N, T> &e = g1.get_elem(it1);
            if (e.get_npart() != it->first) continue;

            abs_index<N + M> aix(part.get_pdims());
            do {
                const index<N + M> &ix1 = aix.get_index();

                index<N> i1;
                for (size_t i = 0, j = 0; i < N + M; i++)
                    if (msk1[i]) { i1[j] = ix1[i]; j++; }
                i1.permute(p1inv);

                if (e.is_forbidden(i1)) {
                    part.mark_forbidden(ix1);
                    continue;
                }

                index<N> i2 = e.get_direct_map(i1);
                if (i1 == i2) continue;

                index<N + M> ix2;
                for (size_t i = 0, j = 0; i < N + M; i++)
                    if (msk1[i]) { ix2[i] = i2[j]; j++; }
                    else { ix2[i] = ix1[i]; }

                if (part.is_forbidden(ix1))
                    part.mark_forbidden(ix2);
                else if (part.is_forbidden(ix2))
                    part.mark_forbidden(ix1);
                else
                    part.add_map(ix1, ix2, e.get_sign(i1, i2));

            } while (aix.inc());
        }
        for(typename adapter2_t::iterator it2 = g2.begin();
                it2 != g2.end(); it2++) {

            const se_part<M, T> &e = g2.get_elem(it2);
            if (e.get_npart() != it->first) continue;

            abs_index<N + M> aix(part.get_pdims());
            do {
                const index<N + M> &ix1 = aix.get_index();

                index<M> i1;
                for (size_t i = 0, j = 0; i < N + M; i++)
                    if (msk2[i]) { i1[j] = ix1[i]; j++; }
                i1.permute(p2inv);

                if (e.is_forbidden(i1)) {
                    part.mark_forbidden(ix1);
                    continue;
                }

                index<M> i2 = e.get_direct_map(i1);
                if (i1 == i2) continue;

                index<N + M> ix2;
                for (size_t i = 0, j = 0; i < N + M; i++)
                    if (msk2[i]) { ix2[i] = i2[j]; j++; }
                    else { ix2[i] = ix1[i]; }

                if (part.is_forbidden(ix1))
                    part.mark_forbidden(ix2);
                else if (part.is_forbidden(ix2))
                    part.mark_forbidden(ix1);
                else
                    part.add_map(ix1, ix2, e.get_sign(i1, i2));

            } while (aix.inc());
        }

        params.g3.insert(part);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_DIRPROD_IMPL_PART_H
