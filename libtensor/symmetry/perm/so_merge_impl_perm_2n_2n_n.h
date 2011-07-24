#ifndef LIBTENSOR_SO_MERGE_IMPL_PERM_2N_2N_N_H
#define LIBTENSOR_SO_MERGE_IMPL_PERM_2N_2N_N_H

namespace libtensor {

/** \brief Implementation of so_merge<2 N, 2 N, N, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> > :
public
symmetry_operation_impl_base< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<2 * N, 2 * N, N, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

template<size_t N, typename T>
const char *
symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> >
::k_clazz =
    "symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> >";

template<size_t N, typename T>
void
symmetry_operation_impl< so_merge<2 * N, 2 * N, N, T>, se_perm<N, T> >
::do_perform(symmetry_operation_params_t &params) const {

	static const char *method =
		"do_perform(symmetry_operation_params_t&)";

	//	Adapter type for the input group
	//
    typedef se_perm<2 * N, T> el1_t;
	typedef symmetry_element_set_adapter<2 * N, T, el1_t> adapter1_t;
	typedef symmetry_element_set_adapter<N - M, T, element_t> adapter_t;

	//	Verify that the projection mask is correct
	//
    size_t nm = 0;
    mask<2 * N> tm, mm;
	for (size_t k = 0; k < N; k++) {
	    const mask<2 * N> &m = params.msk[k];

	    bool found = false;
	    for(size_t i = 0; i < 2 * N; i++) {
	        if(! m[i]) continue;
	        if (tm[i]) {
	            throw bad_parameter(g_ns, k_clazz, method,
	                    __FILE__, __LINE__, "Masks overlap.");
	        }

	        if (found) mm[i] = true;
	        else found = true;

	        tm[i] = true;
	        nm++;
	    }
	}

	if(nm != 2 * N) {
		throw bad_parameter(g_ns, k_clazz, method, __FILE__, __LINE__,
			"params.msk");
	}

	adapter1_t g1(params.grp1);
	permutation_group<2 * N, T> grp1(g1);

	// Create permutation group that stabilizes all vanishing indexes (at the
	// same time)
	permutation_group<N, T> grp2;
	grp1.stabilize(mm, grp2);

	symmetry_element_set<N, T> set2(element_t::k_sym_type);
	grp.convert(set2);

	// Extend the generating set
	for (size_t k = 0; k < N; k++) {


	}
	// 2) Permutation group of merged indexes

	// 3) Permutation group of merged indexes that remain

	// 4) Extend the latter to a group of merged indexes

	// 5) Compute the intersection of groups from step 2) and 4)

	symmetry_element_set<N - M, T> set(se_perm<N - M, T>::k_sym_type);
	groupx.convert(set);

	adapterx_t gx(set);

	for (typename adapterx_t::iterator it = gx.begin(); it != gx.end(); it++) {

		const elx_t &ex = gx.get_elem(it);

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
		se_perm<N - M + 1, T> e2(pb.get_perm(), ex.is_symm());
		params.grp2.insert(e2);

	}
}


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_IMPL_PERM_H
