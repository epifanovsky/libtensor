#ifndef LIBTENSOR_SO_REDUCE_SE_PART_H
#define LIBTENSOR_SO_REDUCE_SE_PART_H

#include <libtensor/core/dimensions.h>
#include "se_part.h"
#include "so_reduce.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {


/** \brief Implementation of so_reduce<N, T> for se_part<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_reduce<N, M, T>, se_part<N - M, T> > :
public symmetry_operation_impl_base< so_reduce<N, M, T>, se_part<N - M, T> > {
public:
    static const char *k_clazz; //!< Class name

    enum {
        NA = N,     //!< Dimension of input symmetry
        NB = N - M, //!< Dimension of result symmetry
        NR = M      //!< Dimension of reduction
    };

public:
    typedef so_reduce<N, M, T> op_t;
    typedef se_part<NA, T> ela_t;
    typedef se_part<NB, T> elb_t;
    typedef symmetry_operation_params<op_t> symmetry_operation_params_t;

public:
    virtual ~symmetry_operation_impl() { }

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

private:
    static bool is_forbidden(const ela_t &el,
            const index<NA> &idx, const dimensions<NA> &subdims);
    static bool map_exists(const ela_t &el, const index<NA> &ia,
            const index<NA> &ib, const dimensions<NA> &subdims);
};

} // namespace libtensor

#endif // LIBTENSOR_SO_REDUCE_SE_PART_H
