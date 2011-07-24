#ifndef LIBTENSOR_SO_MERGE_IMPL_PERM_N_N_1_H
#define LIBTENSOR_SO_MERGE_IMPL_PERM_N_N_1_H

namespace libtensor {

/** \brief Implementation of so_merge<N, N, 1, T> for se_perm<1, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<1, T> > :
    public symmetry_operation_impl_base<
        so_merge<N, N, 1, T>, se_perm<1, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, N, 1, T> operation_t;
    typedef se_perm<1, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, typename T>
const char *symmetry_operation_impl<
    so_merge<N, N, 1, T>, se_perm<1, T> >::k_clazz =
    "symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<1, T> >";

template<size_t N, typename T>
void symmetry_operation_impl< so_merge<N, N, 1, T>,
	se_perm<1, T> >::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk;
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_N_N_1_H
