#ifndef LIBTENSOR_SO_MERGE_SE_PERM_H
#define LIBTENSOR_SO_MERGE_SE_PERM_H

#include "se_perm.h"
#include "so_merge.h"
#include "symmetry_element_set_adapter.h"
#include "symmetry_operation_impl_base.h"

namespace libtensor {

/**	\brief Implementation of so_merge<N, M, K, T> for se_perm<N - M + K, T>
	\tparam N Tensor order.
	\tparam T Tensor element type.

	\ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, size_t K, typename T>
class symmetry_operation_impl< so_merge<N, M, K, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_merge<N, M, K, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, M, K, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
        symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

/** \brief Implementation of so_merge<N, N, 1, T> for se_perm<1, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, typename T>
class symmetry_operation_impl< so_merge<N, N, 1, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_merge<N, N, 1, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, N, 1, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t>
    symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const;

};

/** \brief Implementation of so_merge<N, M, M, T> for se_perm<N, T>
    \tparam N Tensor order.
    \tparam T Tensor element type.

    \ingroup libtensor_symmetry
 **/
template<size_t N, size_t M, typename T>
class symmetry_operation_impl< so_merge<N, M, M, T>, se_perm<N, T> > :
public symmetry_operation_impl_base< so_merge<N, M, M, T>, se_perm<N, T> > {

public:
    static const char *k_clazz; //!< Class name

public:
    typedef so_merge<N, M, M, T> operation_t;
    typedef se_perm<N, T> element_t;
    typedef symmetry_operation_params<operation_t> symmetry_operation_params_t;

protected:
    virtual void do_perform(symmetry_operation_params_t &params) const {
        for (typename symmetry_element_set<N, T>::const_iterator it =
                params.grp1.begin(); it != params.grp1.end(); it++) {
            params.grp2.insert(params.grp1.get_elem(it));
        }
    }

};


} // namespace libtensor

#endif // LIBTENSOR_SO_MERGE_SE_PERM_H
