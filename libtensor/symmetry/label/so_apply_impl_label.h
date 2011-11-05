#ifndef LIBTENSOR_SO_APPLY_IMPL_LABEL_H
#define LIBTENSOR_SO_APPLY_IMPL_LABEL_H

#include "../../defs.h"
#include "../../exception.h"
#include "../symmetry_element_set_adapter.h"
#include "../symmetry_operation_impl_base.h"
#include "../so_apply.h"
#include "../se_label.h"

namespace libtensor {


/**	\brief Implementation of so_apply<N, T> for se_label<N, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	The current implementation of so_apply<N, T> for se_label<N, T> permutes
	every symmetry element if necessary and sets all target labels in the
	result.

	\ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_apply<N, T>, se_label<N, T> > :
public symmetry_operation_impl_base< so_apply<N, T>, se_label<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_apply<N, T> operation_t;
    typedef se_label<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;
};


template<size_t N, typename T>
const char *symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >::k_clazz =
        "symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >";


template<size_t N, typename T>
void symmetry_operation_impl< so_apply<N, T>, se_label<N, T> >::do_perform(
        symmetry_operation_params_t &params) const {

    static const char *method =
            "do_perform(const symmetry_operation_params_t&)";

    typedef symmetry_element_set_adapter< N, T, se_label<N, T> > adapter_t;
    adapter_t g1(params.grp1);
    params.grp2.clear();

    for (typename adapter_t::iterator it1 = g1.begin();
            it1 != g1.end(); it1++) {

        se_label<N, T> e2(g1.get_elem(it1));
        e2.permute(params.perm1);

        e2.delete_target();
        for (typename element_t::label_t l = 0;
                l < e2.get_n_labels(); l++) e2.add_target(l);

        params.grp2.insert(e2);
    }
}


} // namespace libtensor

#endif // LIBTENSOR_SO_APPLY_IMPL_LABEL_H
