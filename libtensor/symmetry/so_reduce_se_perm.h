#ifndef LIBTENSOR_SO_REDUCE_SE_PERM_H
#define LIBTENSOR_SO_REDUCE_SE_PERM_H

#include "se_perm.h"
#include "so_reduce.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/** \brief Implementation of so_reduce<N, M, K, T> for se_perm<N, T>
    \tparam N Input tensor order.
    \tparam M Reduced dimensions.
    \tparam K Reduction steps.
    \tparam T Tensor element type.

    The implementation reduces masked dimensions setwise.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t NM, typename T>
class symmetry_operation_impl< so_reduce<N, M, T>, se_perm<NM, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, T>, se_perm<NM, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order1 = N; //!< Dimension of input
    static const size_t k_order2 = N - M; //!< Dimension of result

public:
    typedef so_reduce<N, M, T> op_t;
    typedef se_perm<k_order1, T> el1_t;
    typedef se_perm<k_order2, T> el2_t;
    typedef symmetry_operation_params<op_t> symmetry_operation_params_t;

public:
    virtual ~symmetry_operation_impl() { }

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PERM_H
