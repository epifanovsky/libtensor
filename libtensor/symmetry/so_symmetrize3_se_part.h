#ifndef LIBTENSOR_SO_SYMMETRIZE3_SE_PART_H
#define LIBTENSOR_SO_SYMMETRIZE3_SE_PART_H

#include "se_part.h"
#include "so_symmetrize3.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/**	\brief Implementation of so_symmetrize3<N, T> for se_part<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_symmetrize3<N, T>, se_part<N, T> > :
public symmetry_operation_impl_base< so_symmetrize3<N, T>, se_part<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_symmetrize3<N, T> operation_t;
    typedef se_part<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};

} // namespace libtensor

#endif // LIBTENSOR_SO_SYMMETRIZE3_SE_PART_H
