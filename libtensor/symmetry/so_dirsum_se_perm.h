#ifndef LIBTENSOR_SO_DIRSUM_SE_PERM_H
#define LIBTENSOR_SO_DIRSUM_SE_PERM_H

#include "se_perm.h"
#include "so_dirsum.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"
#include "permutation_group.h"

namespace libtensor {

/** \brief Implementation of so_dirsum<N, M, T> for se_perm<N + M, T>
    \tparam N Tensor order 1.
    \tparam M Tensor order 2
    \tparam T Tensor element type.
    \tparam CGT Cyclic group traits

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t NM, typename T>
class symmetry_operation_impl< so_dirsum<N, M, T>, se_perm<NM, T> > :
public symmetry_operation_impl_base< so_dirsum<N, M, T>, se_perm<NM, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_dirsum<N, M, T> operation_t;
    typedef se_perm<N + M, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

public:
    virtual ~symmetry_operation_impl() { }

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

private:
    static void combine(const symmetry_element_set<N, T> &set1,
            const permutation<M> &p2, const scalar_transf<T> &tr2,
            permutation_group<N + M, T> &grp);

    static void combine(const permutation<N> &p1, const scalar_transf<T> &tr1,
            const symmetry_element_set<M, T> &set2,
            permutation_group<N + M, T> &grp);

    static size_t lcm(const std::vector<size_t> &seq);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_DIRSUM_SE_PERM_H
