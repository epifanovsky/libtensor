#ifndef LIBTENSOR_SO_MERGE_SE_LABEL_H
#define LIBTENSOR_SO_MERGE_SE_LABEL_H

#include "se_label.h"
#include "so_merge.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {


/**	\brief Implementation of so_merge<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam M Decrement in tensor order
	\tparam T Tensor element type.

	This implementation sets the target label to all labels.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_merge<N, M, T>, se_label<N - M, T> > :
public symmetry_operation_impl_base< so_merge<N, M, T>, se_label<N - M, T> > {

public:
    static const char *k_clazz; //!< Class name
    static const size_t k_order1 = N; //!< Dimension of input
    static const size_t k_order2 = N - M; //!< Dimension of result

public:
    typedef so_merge<N, M, T> operation_t;
    typedef se_label<N, T> el1_t;
    typedef se_label<N - M, T> el2_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_LABEL_H
