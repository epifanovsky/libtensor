#ifndef LIBTENSOR_SO_MERGE_IMPL_PERM_N_M_1_H
#define LIBTENSOR_SO_MERGE_IMPL_PERM_N_M_1_H

namespace libtensor {

/** \brief Implementation of so_merge<N, M, 1, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> > :
    public symmetry_operation_impl_base<
        so_merge<N, M, 1, T>, se_perm<N - M + 1, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, M, 1, T> operation_t;
    typedef se_perm<N - M + 1, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, size_t M, typename T>
const char *
symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> >
::k_clazz =
    "symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> >";

template<size_t N, size_t M, typename T>
void symmetry_operation_impl< so_merge<N, M, 1, T>, se_perm<N - M + 1, T> >
::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
    typedef se_perm<N, T> el1_t;
    typedef se_perm<N - M, T> el2_t;
	typedef symmetry_element_set_adapter<N, T, el1_t> adapter1_t;
	typedef symmetry_element_set_adapter<N - M, T, el2_t> adapter2_t;

	//	Verify that the projection mask is correct
	//
	const mask<N> &m = params.msk[0];
	size_t nm = 0;
	for(size_t i = 0; i < N; i++) if(m[i]) nm++;

	if(nm != M) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	adapter1_t g1(params.grp1);
	permutation_group<N, T> grp1(g1);
	permutation_group<N - M, T> grpx;
	group1.stabilize(params.msk, grpx);

	symmetry_element_set<N - M, T> set(el2_t::k_sym_type);
	grpx.convert(set);

	adapter2_t g2(set);

	for (typename adapter2_t::iterator it = g2.begin(); it != g2.end(); it++) {

		const el2_t &e2 = g2.get_elem(it);

		//	Projects the permutations onto a larger
		//	space and form a symmetry element
		sequence<N - M, size_t> a1(0);
		sequence<N - M + 1, size_t> a2a(0), a2b(0);
		for (size_t j = 0; j < N - M; j++) a1[j] = j;
		ex.get_perm().apply(a1);

		bool done = false;
		for(size_t j = 0, k = 0, l = 0; j < N; j++) {
			if (m[j]) {
				if (! done) {
					a2a[k] = N - M;
					a2b[k] = N - M;
					k++;
					done = true;
				}
				continue;
			}

			a2a[k] = l;
			a2b[k] = a1[l];
			k++;
			l++;
		}

		permutation_builder<N - M + 1> pb(a2b, a2a);
		element_t el(pb.get_perm(), e2.is_symm());
		params.grp2.insert(el);
	}
}

} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_N_M_1_ H
