#ifndef LIBTENSOR_SO_MERGE_SE_PART_H
#define LIBTENSOR_SO_MERGE_SE_PART_H

#include <libtensor/core/dimensions.h>
#include "se_part.h"
#include "so_merge.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/** \brief Implementation of so_merge<N, M, T> for se_part<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    The implementation merges the masked dimensions together.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t NM, typename T>
class symmetry_operation_impl< so_merge<N, M, T>, se_part<NM, T> > :
public symmetry_operation_impl_base< so_merge<N, M, T>, se_part<NM, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order1; //!< Order of input
    static const size_t k_order2; //!< Order of result

public:
    typedef so_merge<N, M, T> operation_t;
    typedef se_part<N, T> el1_t;
    typedef se_part<N - M, T> el2_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

private:
    static bool is_forbidden(const el1_t &el,
            const index<N> &idx, const dimensions<N> &subdims);
    static bool map_exists(const el1_t &el, const index<N> &ia,
            const index<N> &ib, const dimensions<N> &subdims);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_PERM_H
