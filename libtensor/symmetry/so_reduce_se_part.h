#ifndef LIBTENSOR_SO_REDUCE_SE_PART_H
#define LIBTENSOR_SO_REDUCE_SE_PART_H

#include <libtensor/core/dimensions.h>
#include "se_part.h"
#include "so_reduce.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {


/**	\brief Implementation of so_reduce<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, T>, se_part<N - M, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order1 = N; //!< Dimension of input
    static const size_t k_order2 = N - M; //!< Dimension of result

public:
    typedef so_reduce<N, M, T> op_t;
    typedef se_part<k_order1, T> el1_t;
    typedef se_part<k_order2, T> el2_t;
    typedef symmetry_operation_params<op_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

private:
    static bool is_forbidden(const el1_t &el,
            const index<k_order1> &idx, const dimensions<k_order1> &subdims);
    static bool map_exists(const el1_t &el, const index<k_order1> &ia,
            const index<k_order1> &ib, const dimensions<k_order1> &subdims);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PART_H
